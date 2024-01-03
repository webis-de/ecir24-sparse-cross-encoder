import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Literal, Optional, Tuple, Union

import ir_datasets
import lightning.pytorch as pl
import pandas as pd
import torch
import torch.utils.data
import transformers

from .ir_dataset_utils import get_base
from .ir_dataset_utils import load as load_ir_dataset


def _read_ir_dataset(
    dataset: ir_datasets.Dataset,
    data_type: Union[Literal["queries"], Literal["qrels"], Literal["scoreddocs"]],
) -> pd.DataFrame:
    # TODO use pd.read_csv to load queries etc faster, see evaluate-sparse_cross_encoder.ipybn
    if data_type == "queries":
        if not dataset.has_queries():
            raise ValueError(f"dataset {dataset.dataset_id()} does not have queries")
        data = pd.DataFrame(dataset.queries_iter())
        data = data.loc[:, ["query_id", "text"]]
        data["text"] = data["text"].fillna("").str.strip()
        if data.dtypes["query_id"] != object or data.dtypes["text"] != object:
            data = data.astype({"query_id": str, "text": str})
    elif data_type == "qrels":
        if not dataset.has_qrels():
            raise ValueError(f"dataset {dataset.dataset_id()} does not have qrels")
        data = pd.DataFrame(dataset.qrels_iter())
        data = data.loc[:, ["query_id", "doc_id", "relevance"]]
        if data.dtypes["query_id"] != object or data.dtypes["doc_id"] != object:
            data = data.astype({"query_id": str, "doc_id": str})
    elif data_type == "scoreddocs":
        if not dataset.has_scoreddocs():
            raise ValueError(f"dataset {dataset.dataset_id()} does not have scoreddocs")
        if hasattr(dataset.scoreddocs_handler(), "df"):
            data = dataset.scoreddocs_handler().df
        else:
            data = pd.DataFrame(dataset.scoreddocs_iter())
        if data.dtypes["query_id"] != object or data.dtypes["doc_id"] != object:
            data = data.astype({"query_id": str, "doc_id": str})
        if "rank" not in data:
            data["rank"] = (
                data.groupby(["query_id", "score"])
                .rank(method="first", ascending=False)
                .astype(int)
            )
        if data["rank"].iloc[0] != 1:
            # heuristic for sorting, if by chance the first rank is 1, bad luck
            data = data.sort_values(["query_id", "rank"])
    else:
        raise ValueError(f"invalid data_type: {data_type}")

    return data


class CacheLoader:
    __instance = None
    __data = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def read_ir_dataset(
        self,
        dataset: ir_datasets.Dataset,
        data_type: Union[Literal["queries"], Literal["qrels"], Literal["scoreddocs"]],
    ) -> pd.DataFrame:
        dataset_id = dataset.dataset_id() + "/" + data_type
        if data_type in ("queries", "qrels"):
            dataset_id = re.sub(r"__.+__", "", dataset_id)
        if CacheLoader.__data is None:
            CacheLoader.__data = {}

        if dataset_id in CacheLoader.__data:
            return CacheLoader.__data[dataset_id]

        # TODO use pd.read_csv to load queries etc faster, see evaluate-sparse_cross_encoder.ipynb
        data = _read_ir_dataset(dataset, data_type)

        CacheLoader.__data[dataset_id] = data
        return data

    def clear(self):
        CACHE_LOADER.__data = None


CACHE_LOADER = CacheLoader()
read_ir_dataset = CACHE_LOADER.read_ir_dataset


class DocStore:
    def __init__(
        self,
        docstore: ir_datasets.indices.Docstore,
        doc_fields: Optional[Iterable[str]],
    ) -> None:
        self.docstore = docstore
        self.doc_fields = doc_fields

    def __getitem__(self, doc_id: str) -> str:
        doc = self.docstore.get(doc_id)
        if self.doc_fields:
            contents = " ".join(
                [
                    getattr(doc, field)
                    for field in self.doc_fields
                    if hasattr(doc, field)
                ]
            )
        else:
            contents = doc.default_text()
        return contents

    @property
    def path(self) -> str:
        if hasattr(self.docstore, "path"):
            return self.docstore.path
        elif hasattr(self.docstore, "_path"):
            return self.docstore._path
        else:
            raise AttributeError("docstore has no path attribute")

    def get(self, doc_id, field=None):
        return self.docstore.get(doc_id, field)

    def get_many(self, doc_ids, field=None):
        return self.docstore.get_many(doc_ids, field)


class IRDataset:
    @staticmethod
    def get_docs(
        ir_dataset: ir_datasets.Dataset,
        doc_fields: Optional[Iterable[str]] = None,
    ) -> DocStore:
        return DocStore(ir_dataset.docs_store(), doc_fields)

    @staticmethod
    def load_queries(
        ir_dataset: ir_datasets.Dataset,
        query_ids: Optional[Iterable[str]] = None,
    ) -> Dict[str, str]:
        queries = read_ir_dataset(ir_dataset, "queries")
        if query_ids is not None:
            queries = queries[queries["query_id"].isin(query_ids)]
        queries = queries.set_index("query_id")["text"].to_dict()
        return queries

    @staticmethod
    def load_run(
        ir_dataset: ir_datasets.Dataset,
    ) -> pd.DataFrame:
        run = read_ir_dataset(ir_dataset, "scoreddocs")
        return run


class RunDataset(torch.utils.data.Dataset, IRDataset):
    def __init__(self, ir_dataset: ir_datasets.Dataset, depth: int):
        self.run = self.load_run(ir_dataset)
        self.groups = self.run.groupby("query_id")
        self.query_ids = list(self.groups.groups.keys())
        self.docs = self.get_docs(ir_dataset)
        self.queries = self.load_queries(ir_dataset, self.query_ids)
        self.depth = depth

    def __len__(self) -> int:
        return len(self.query_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        query_id = self.query_ids[index]
        group = self.groups.get_group(query_id).copy()
        if self.depth != -1:
            valid = group["rank"] <= self.depth
            group = group.loc[valid]
        group = group.drop_duplicates(subset=["doc_id"])
        query = self.queries[query_id]
        doc_ids = list(group["doc_id"])
        docs = [self.docs[doc_id] for doc_id in doc_ids]
        sample = {
            "query": query,
            "docs": docs,
            "doc_ids": doc_ids,
            "query_id": query_id,
        }
        return sample


class DocPairDataset(torch.utils.data.IterableDataset, IRDataset):
    def __init__(
        self,
        ir_dataset: ir_datasets.Dataset,
    ) -> None:
        super().__init__()
        self.ir_dataset = ir_dataset
        self.docs = self.get_docs(ir_dataset)
        self.queries = self.load_queries(ir_dataset)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for (
            query_id,
            pos_doc_id,
            neg_doc_id,
            *scores,
        ) in self.ir_dataset.docpairs_iter():
            query = self.queries[query_id]
            pos_doc = self.docs[pos_doc_id]
            neg_doc = self.docs[neg_doc_id]
            labels = [1, 0]
            if scores:
                labels = [float(score) for score in scores]
            yield {
                "query": query,
                "docs": [pos_doc, neg_doc],
                "labels": labels,
            }


class SparseCrossEncoderDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str,
        ir_dataset_path: Path,
        truncate: bool = True,
        max_query_length: int = 32,
        max_length: int = 512,
        batch_size: int = 1,
        depth: int = 100,
        num_workers: int = 0,
    ):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

        self.ir_dataset_path = ir_dataset_path

        self.truncate = truncate
        self.max_query_length = max_query_length
        self.max_length = max_length
        self.batch_size = batch_size
        self.depth = depth
        self.num_workers = num_workers

        self.train_dataset = None
        self.predict_dataset = None
        super().__init__()

    def setup_train(self, ir_dataset: ir_datasets.Dataset) -> None:
        self.train_dataset = DocPairDataset(ir_dataset)

    def setup_predict(self, ir_dataset: ir_datasets.Dataset) -> None:
        self.predict_dataset = RunDataset(ir_dataset=ir_dataset, depth=self.depth)

    def setup(self, stage: Optional[str] = None):
        dataset = load_ir_dataset(self.ir_dataset_path)
        if stage in ("fit", None):
            self.setup_train(dataset)

        if stage == "predict":
            self.setup_predict(dataset)

        CACHE_LOADER.clear()

    def train_dataloader(self):
        assert self.train_dataset is not None
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_and_tokenize,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        assert self.predict_dataset is not None
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_and_tokenize,
            num_workers=self.num_workers,
        )

    def collate_and_tokenize(self, batch: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        query_input_ids = self.tokenizer(
            [batch_dict["query"] for batch_dict in batch],
            truncation=self.truncate,
            max_length=self.max_query_length if self.truncate else None,
        ).input_ids
        # remove [CLS] token
        query_input_ids = [torch.tensor(input_ids[1:]) for input_ids in query_input_ids]
        max_query_length = max(tensor.shape[0] for tensor in query_input_ids)
        max_doc_length = self.max_length - max_query_length
        doc_input_ids = []
        for batch_dict in batch:
            batch_doc_input_ids = self.tokenizer(
                batch_dict["docs"],
                truncation=self.truncate,
                max_length=max_doc_length if self.truncate else None,
            ).input_ids
            # remove [CLS] token
            batch_doc_input_ids = [
                torch.tensor(input_ids[1:]) for input_ids in batch_doc_input_ids
            ]
            doc_input_ids.append(batch_doc_input_ids)
        collated_batch = defaultdict(list)
        collated_batch["query_input_ids"] = query_input_ids
        collated_batch["doc_input_ids"] = doc_input_ids
        for batch_dict in batch:
            for key, value in batch_dict.items():
                if "input_ids" in key:
                    continue
                if "labels" in key:
                    value = torch.tensor(value).float()
                    collated_batch[key].append(value)
                if key in ("doc_ids", "query_id"):
                    collated_batch[key].append(value)
        return dict(collated_batch)
