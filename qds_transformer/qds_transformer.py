import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.absolute()
sys.path.append(root_dir.as_posix())

from typing import Any, Dict, Iterable

import nltk
import torch
from nltk import sent_tokenize
from transformers import AutoTokenizer

from sparse_cross_encoder.data.datamodule import SparseCrossEncoderDataModule
from sparse_cross_encoder.model import loss_utils
from sparse_cross_encoder.model.sparse_cross_encoder import (
    AttentionMasks,
    HiddenStates,
    SparseCrossEncoderConfig,
)
from sparse_cross_encoder.model.sparse_cross_encoder_module import (
    SparseCrossEncoderModule,
)
from main import SparseCrossEncoderLightningCLI

nltk.download("punkt")


class QDSTransformerDataModule(SparseCrossEncoderDataModule):
    def __init__(
        self,
        model_name_or_path: str,
        ir_dataset_paths: Iterable[str | Path],
        truncate: bool = True,
        max_query_length: int = 32,
        max_length: int = 512,
        batch_size: int = 1,
        depth: int = 100,
        num_workers: int = 0,
    ):
        super().__init__(
            model_name_or_path,
            ir_dataset_paths,
            truncate,
            max_length,
            max_query_length,
            batch_size,
            depth,
            num_workers,
        )
        if "[SOS]" not in self.tokenizer.vocab:
            self.tokenizer.add_tokens(["[SOS]"])

    def collate_and_tokenize(self, batch: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        for sample in batch:
            batch_docs = []
            for doc in sample["docs"]:
                sentences = sent_tokenize(doc)
                # merge sentences with less than 5 tokens
                cleaned_sentences = []
                cleaned_sentence = []
                for sentence in sentences:
                    tokens = self.tokenizer(
                        sentence, add_special_tokens=False
                    ).input_ids
                    if len(tokens) < 5:
                        cleaned_sentence.append(sentence)
                    else:
                        cleaned_sentences.append(
                            " ".join(cleaned_sentence + [sentence])
                        )
                        cleaned_sentence = []
                if len(cleaned_sentence) > 0:
                    cleaned_sentences.append(" ".join(cleaned_sentence))
                sentences = ["[SOS] " + sentence for sentence in sentences]
                batch_docs.append(" ".join(sentences))
            sample["docs"] = batch_docs
        return super().collate_and_tokenize(batch)


class QDSTransformerModule(SparseCrossEncoderModule):
    def __init__(
        self,
        model_name_or_path: str,
        config: SparseCrossEncoderConfig | None = None,
        loss_function: loss_utils.LossFunc = loss_utils.RankNet(),
        compile_model: bool = True,
        qds_transformer: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name_or_path,
            config,
            loss_function,
            compile_model,
            **kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if "[SOS]" in tokenizer.vocab:
            sos_token_id = tokenizer.vocab["[SOS]"]
        else:
            sos_token_id = self.config.vocab_size
            self.sparse_cross_encoder.resize_token_embeddings(
                self.config.vocab_size + 1
            )
        self.sparse_cross_encoder.bert.get_hidden_states = hidden_states_wrapper(
            self.sparse_cross_encoder.bert.get_hidden_states, sos_token_id
        )
        self.sparse_cross_encoder.bert.get_attention_masks = attention_mask_wrapper(
            self.sparse_cross_encoder.bert.get_attention_masks, sos_token_id
        )
        self.oom_batches = 0
        self.qds_transformer = qds_transformer

    def on_train_start(self) -> None:
        self.oom_batches = 0

    def on_train_end(self) -> None:
        print("**************************")
        print(f"OOM batches: {self.oom_batches}")
        print("**************************")

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        try:
            loss = super().training_step(batch, batch_idx)
        except torch.cuda.OutOfMemoryError:
            loss = torch.tensor(0.0).to(self.device).requires_grad_()
            self.oom_batches += 1
            self.log("oom_batches", float(self.oom_batches), prog_bar=True)
        return loss


def hidden_states_wrapper(get_hidden_states, sos_token_id: int):
    def wrapped_get_hidden_states(input_tensors: HiddenStates) -> HiddenStates:
        hidden_states = get_hidden_states(input_tensors)
        sos_masks = input_tensors.doc == sos_token_id
        max_sentences = sos_masks.sum(dim=-1).max()
        min_sentences = sos_masks.sum(dim=-1).min()
        new_doc_hidden_states = torch.zeros_like(hidden_states.doc)
        new_query_hidden_states = torch.zeros(
            hidden_states.query.shape[0],
            hidden_states.query.shape[1],
            hidden_states.query.shape[2] + max_sentences,
            hidden_states.query.shape[3],
        ).to(hidden_states.query)
        new_query_hidden_states[
            :, :, : hidden_states.query.shape[2]
        ] = hidden_states.query
        for batch_idx, (
            query_hidden_states,
            doc_hidden_states,
            sos_mask,
        ) in enumerate(zip(hidden_states.query, hidden_states.doc, sos_masks)):
            sos_hidden_states = doc_hidden_states[0, sos_mask[0]]
            _new_doc_hidden_states = doc_hidden_states[0, ~sos_mask[0]]
            new_doc_hidden_states[
                batch_idx, 0, : _new_doc_hidden_states.shape[0]
            ] = _new_doc_hidden_states
            new_query_hidden_states[
                batch_idx,
                0,
                query_hidden_states.shape[1] : (
                    query_hidden_states.shape[1] + sos_hidden_states.shape[0]
                ),
            ] = sos_hidden_states
        if min_sentences:
            new_doc_hidden_states = new_doc_hidden_states[:, :, :-min_sentences]
        hidden_states.doc = new_doc_hidden_states
        hidden_states.query = new_query_hidden_states
        return hidden_states

    return wrapped_get_hidden_states


def attention_mask_wrapper(get_attention_masks, sos_token_id: int):
    def wrapped_get_attention_masks(input_tensors: HiddenStates) -> AttentionMasks:
        attention_masks = get_attention_masks(input_tensors)
        sos_masks = input_tensors.doc == sos_token_id
        max_sentences = sos_masks.sum(dim=-1).max()
        min_sentences = sos_masks.sum(dim=-1).min()
        new_doc_attention_masks = attention_masks.doc.clone()
        for batch_idx, sos_mask in enumerate(sos_masks):
            num_sentences = sos_mask.sum()
            num_masked = (new_doc_attention_masks[batch_idx, 0, 0, 0, :] < 0).sum()
            new_doc_attention_masks[
                batch_idx, 0, 0, 0, -num_masked - num_sentences :
            ] = torch.finfo(new_doc_attention_masks.dtype).min
        if min_sentences:
            new_doc_attention_masks = new_doc_attention_masks[..., :-min_sentences]
        new_query_attention_masks = torch.zeros(
            *attention_masks.query.shape[:-1],
            attention_masks.query.shape[-1] + max_sentences,
        ).to(attention_masks.query)
        new_query_attention_masks[
            ..., : attention_masks.query.shape[-1]
        ] = attention_masks.query
        sos_attention_mask = (
            torch.arange(max_sentences, device=sos_masks.device) >= sos_masks.sum(-1)
        ).int() * torch.finfo(attention_masks.query.dtype).min
        new_query_attention_masks[
            :, 0, 0, 0, attention_masks.query.shape[-1] :
        ] = sos_attention_mask
        attention_masks.query = new_query_attention_masks
        attention_masks.doc = new_doc_attention_masks
        return attention_masks

    return wrapped_get_attention_masks


def main():
    """
    generate config using `python main.py fit --print_config > config.yaml`
    additional callbacks at:
    https://lightning.ai/docs/pytorch/stable/api_references.html#callbacks

    Example:
        To obtain a default config:

            python main.py fit \
                --trainer.callbacks=ModelCheckpoint \
                --optimizer AdamW \
                --trainer.logger WandbLogger \
                --lr_scheduler LinearSchedulerWithWarmup \
                --print_config > default.yaml

        To run with the default config:

            python main.py fit \
                --config default.yaml

    """
    SparseCrossEncoderLightningCLI(
        QDSTransformerModule,
        QDSTransformerDataModule,
        save_config_kwargs={"config_filename": "pl_config.yaml", "overwrite": True},
    )


if __name__ == "__main__":
    main()
