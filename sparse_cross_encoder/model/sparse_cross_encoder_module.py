from typing import Any, Dict, List, Optional, Sequence
from pathlib import Path

import lightning.pytorch as pl
import torch
import transformers

from sparse_cross_encoder.model import loss_utils
from sparse_cross_encoder.model.sparse_cross_encoder import (
    SparseCrossEncoderConfig,
    SparseCrossEncoderModelForSequenceClassification,
    SparseCrossEncoderPreTrainedModel,
)


def batch_input(*args, batch_size=1):
    num_batches = len(args[0]) // batch_size
    for i in range(num_batches):
        yield [arg[i * batch_size : (i + 1) * batch_size] for arg in args]
    if len(args[0]) % batch_size != 0:
        yield [arg[num_batches * batch_size :] for arg in args]


class SparseCrossEncoderModule(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        config: Optional[SparseCrossEncoderConfig] = None,
        loss_function: loss_utils.LossFunc = loss_utils.RankNet(),
        compile_model: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.loss_function = loss_function
        base_config = transformers.AutoConfig.from_pretrained(model_name_or_path)

        # monkey patch model_type
        SparseCrossEncoderModelForSequenceClassification.base_model_prefix = (
            base_config.model_type
        )
        SparseCrossEncoderPreTrainedModel.base_model_prefix = base_config.model_type
        SparseCrossEncoderConfig.model_type = base_config.model_type

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        model_config = SparseCrossEncoderConfig.from_pretrained(model_name_or_path)
        if config is not None:
            model_config.update(config.get_diff_dict())
        # model_config.model_type = base_config.model_type
        pad_token_id = tokenizer.pad_token_id
        cls_token_id = (
            tokenizer.cls_token_id if tokenizer.cls_token_id else tokenizer.bos_token_id
        )
        sep_token_id = tokenizer.sep_token_id
        assert pad_token_id is not None
        assert cls_token_id is not None
        assert sep_token_id is not None

        self.config = model_config
        self.config.pad_token_id = pad_token_id
        self.config.cls_token_id = cls_token_id
        self.config.sep_token_id = sep_token_id

        self.sparse_cross_encoder = (
            SparseCrossEncoderModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                config=model_config,
                ignore_mismatched_sizes=True,
            )
        )

        if compile_model:
            torch.compile(self.sparse_cross_encoder)

    def forward(
        self,
        query_input_ids: Sequence[torch.Tensor],
        doc_input_ids: Sequence[Sequence[torch.Tensor]],
    ) -> List[torch.Tensor]:
        out = self.sparse_cross_encoder.forward(query_input_ids, doc_input_ids)
        return out.logits

    def training_step(self, batch, batch_idx):
        query_input_ids = batch["query_input_ids"]
        doc_input_ids = batch["doc_input_ids"]
        logits = self.forward(query_input_ids, doc_input_ids)

        logits = torch.nn.utils.rnn.pad_sequence(
            logits, batch_first=True, padding_value=loss_utils.PAD_VALUE
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            batch["labels"], batch_first=True, padding_value=loss_utils.PAD_VALUE
        )
        loss = self.loss_function.compute(logits, labels)

        self.log("loss", loss, prog_bar=True)
        return loss

    def parse_max_p(
        self,
        query_input_ids: Sequence[torch.Tensor],
        doc_input_ids: Sequence[Sequence[torch.Tensor]],
    ) -> List[List[torch.Tensor]]:
        max_length = max(
            [
                len(doc)
                for batch_doc_input_ids in doc_input_ids
                for doc in batch_doc_input_ids
            ]
        )
        if max_length <= self.config.max_position_embeddings:
            return doc_input_ids
        max_query_length = max(len(query) for query in query_input_ids)
        max_input_length = self.config.max_position_embeddings - max_query_length - 1
        chunk_starts = range(0, max_length - max_input_length, max_input_length // 2)
        new_doc_input_ids = []
        for batch_doc_input_ids in doc_input_ids:
            new_batch_doc_input_ids = []
            for input_ids in batch_doc_input_ids:
                for chunk_start in chunk_starts:
                    new_batch_doc_input_ids.append(
                        input_ids[chunk_start : chunk_start + max_input_length]
                    )
            new_doc_input_ids.append(new_batch_doc_input_ids)
        return new_doc_input_ids

    def compute_max_p(
        self,
        query_input_ids: Sequence[torch.Tensor],
        doc_input_ids: Sequence[Sequence[torch.Tensor]],
        logits: Sequence[torch.Tensor],
    ) -> List[torch.Tensor]:
        max_length = max(
            [
                len(doc)
                for batch_doc_input_ids in doc_input_ids
                for doc in batch_doc_input_ids
            ]
        )
        if max_length <= self.config.max_position_embeddings:
            return logits
        max_query_length = max(len(query) for query in query_input_ids)
        max_input_length = self.config.max_position_embeddings - max_query_length - 1
        chunk_starts = range(0, max_length - max_input_length, max_input_length // 2)
        num_chunks = len(chunk_starts)
        new_logits = []
        for batch_idx, batch_logits in enumerate(logits):
            assert len(batch_logits) % num_chunks == 0
            new_batch_logits = []
            for doc_idx, chunk_idx in enumerate(
                range(0, len(batch_logits), num_chunks)
            ):
                chunk_slice = slice(chunk_idx, chunk_idx + num_chunks)
                doc_logits = batch_logits[chunk_slice]
                pad_bool = torch.tensor(chunk_starts) < len(
                    doc_input_ids[batch_idx][doc_idx]
                )
                doc_logits = doc_logits[pad_bool]
                new_batch_logits.append(doc_logits.max())
            new_logits.append(torch.tensor(new_batch_logits))
        return new_logits

    def predict_step(
        self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> List[torch.Tensor]:
        query_input_ids = batch["query_input_ids"]
        doc_input_ids = batch["doc_input_ids"]
        doc_input_ids = self.parse_max_p(query_input_ids, doc_input_ids)
        logits = self.forward(query_input_ids, doc_input_ids)
        logits = self.compute_max_p(
            batch["query_input_ids"], batch["doc_input_ids"], logits
        )
        return logits

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.trainer is not None and self.trainer.log_dir is not None:
            step = self.trainer.global_step
            self.sparse_cross_encoder.config.save_step = step
            log_dir = Path(self.trainer.log_dir)
            save_path = log_dir / "huggingface_checkpoint"
            self.sparse_cross_encoder.save_pretrained(save_path)
            try:
                self.trainer.datamodule.tokenizer.save_pretrained(save_path)
            except:
                pass
