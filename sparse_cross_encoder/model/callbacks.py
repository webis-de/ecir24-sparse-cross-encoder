import re
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import ir_datasets
import pandas as pd
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter, Callback
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader

from sparse_cross_encoder.data.ir_dataset_utils import load as load_ir_dataset
from sparse_cross_encoder.data.ir_dataset_utils import DASHED_DATASET_MAP

RUN_HEADER = ["query", "q0", "docid", "rank", "score", "system"]


class PredictionWriter(BasePredictionWriter):
    def __init__(self, overwrite: bool = False) -> None:
        super().__init__("batch")
        self.overwrite = overwrite

    def get_run_path(
        self, trainer: Trainer, pl_module: LightningModule, dataset_idx: int
    ) -> Path:
        assert hasattr(trainer, "datamodule")
        assert hasattr(trainer, "ckpt_path")
        ckpt_path = Path(trainer.ckpt_path)
        datamodule = trainer.datamodule
        ir_dataset_path = datamodule.ir_dataset_paths[dataset_idx]
        ir_dataset = load_ir_dataset(ir_dataset_path)
        original_ir_dataset_id = re.sub(
            r"__.+__", "", ir_dataset.dataset_id().split("/")[-1]
        )
        original_ir_dataset = ir_datasets.load(
            DASHED_DATASET_MAP[original_ir_dataset_id]
        )
        dataset_id = original_ir_dataset.dataset_id().replace("/", "-")
        filename = Path(ir_dataset_path).parent.name + "_" + dataset_id + ".run"
        run_file_path = ckpt_path.parent.parent / "runs" / filename
        return run_file_path

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        run_file_path = self.get_run_path(trainer, pl_module, dataloader_idx)
        doc_ids = batch["doc_ids"]
        query_ids = batch["query_id"]
        scores = [float(logit.item()) for logits in prediction for logit in logits]
        query_ids = [
            query_id
            for batch_idx, query_id in enumerate(query_ids)
            for _ in range(len(doc_ids[batch_idx]))
        ]
        doc_ids = [doc_id for doc_ids in doc_ids for doc_id in doc_ids]
        run_df = pd.DataFrame(
            zip(query_ids, doc_ids, scores), columns=["query", "docid", "score"]
        )
        run_df = run_df.sort_values(["query", "score"], ascending=[True, False])
        run_df["rank"] = (
            run_df.groupby("query")["score"]
            .rank(ascending=False, method="first")
            .astype(int)
        )
        run_df["q0"] = 0
        run_df["system"] = "sparse_cross_encoder"
        run_df = run_df[RUN_HEADER]
        run_file_path.parent.mkdir(exist_ok=True)
        if batch_idx == 0:
            mode = "w"
        else:
            mode = "a"
        run_df.to_csv(run_file_path, header=False, index=False, sep="\t", mode=mode)
