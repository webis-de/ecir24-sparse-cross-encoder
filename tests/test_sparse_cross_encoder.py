import pathlib

import pytest
import torch
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from sparse_cross_encoder.model.sparse_cross_encoder import (
    SparseCrossEncoderConfig,
    SparseCrossEncoderModelForSequenceClassification,
)


@pytest.fixture(scope="session")
def bert(model_name: str) -> BertForSequenceClassification:
    return BertForSequenceClassification.from_pretrained(
        model_name, num_hidden_layers=2
    ).eval()


@pytest.fixture(scope="session")
def cross_encoder(
    model_name: str, config_dir: pathlib.Path
) -> SparseCrossEncoderModelForSequenceClassification:
    config_path = config_dir / "cross-encoder.json"
    config = SparseCrossEncoderConfig.from_pretrained(model_name)
    config.update(
        SparseCrossEncoderConfig.from_pretrained(
            config_path, cls_token_id=101, pad_token_id=0, num_hidden_layers=2
        ).get_diff_dict()
    )
    model = SparseCrossEncoderModelForSequenceClassification.from_pretrained(
        model_name, model_type=config.model_type, config=config
    ).eval()
    return model


@pytest.fixture(scope="session")
def sparse_cross_encoder(
    model_name: str, config_dir: pathlib.Path
) -> SparseCrossEncoderModelForSequenceClassification:
    config_path = config_dir / "sparse-cross-encoder.json"
    config = SparseCrossEncoderConfig.from_pretrained(model_name)
    config.update(
        SparseCrossEncoderConfig.from_pretrained(
            config_path, cls_token_id=101, pad_token_id=0, num_hidden_layers=2
        ).get_diff_dict()
    )
    model = SparseCrossEncoderModelForSequenceClassification.from_pretrained(
        model_name, model_type=config.model_type, config=config
    ).eval()
    return model


class TestSparseCrossEncoder:
    def test_load_model(self, model_name: str):
        config = SparseCrossEncoderConfig.from_pretrained(model_name)
        config.sep_token_id = 102
        config.pad_token_id = 0
        sparse_cross_encoder = (
            SparseCrossEncoderModelForSequenceClassification.from_pretrained(
                model_name,
                model_type="bert",
                config=config,
                ignore_mismatched_sizes=True,
            )
        )

        assert sparse_cross_encoder

    def test_load_extended_model(self, model_name: str):
        config = SparseCrossEncoderConfig.from_pretrained(model_name)
        config.sep_token_id = 102
        config.pad_token_id = 0
        config.max_position_embeddings = 4096
        sparse_cross_encoder = (
            SparseCrossEncoderModelForSequenceClassification.from_pretrained(
                model_name,
                model_type="bert",
                config=config,
                ignore_mismatched_sizes=True,
            )
        )

        assert (
            sparse_cross_encoder.bert.embeddings.position_embeddings.num_embeddings
            == 4096
        )

    def test_same_as_sentence_bert(
        self,
        bert: BertForSequenceClassification,
        cross_encoder: SparseCrossEncoderModelForSequenceClassification,
        tokenizer: BertTokenizerFast,
    ):
        query = "How many people live in Berlin?"
        doc = (
            "Berlin has a population of 3,520,031 registered inhabitants "
            "in an area of 891.82 square kilometers."
        )
        cross_encoder_features = tokenizer([query], [doc], return_tensors="pt")
        query_input_ids, doc_input_ids = tokenizer([query, doc]).input_ids
        query_input_ids = [torch.tensor(query_input_ids[1:])]
        doc_input_ids = [[torch.tensor(doc_input_ids[1:])]]

        with torch.no_grad():
            bert_out = bert(**cross_encoder_features, output_hidden_states=True)
            cross_encoder_out = cross_encoder(
                query_input_ids, doc_input_ids, output_hidden_states=True
            )

        for batch_idx, query in enumerate(query_input_ids):
            doc = doc_input_ids[batch_idx][0]
            query_len = len(query)
            doc_len = len(doc)
            for layer_idx in range(len(bert_out.hidden_states)):
                bert_cls = bert_out.hidden_states[layer_idx][batch_idx, 0]
                cross_encoder_cls = cross_encoder_out.hidden_states[layer_idx].cls[
                    batch_idx, 0, 0
                ]
                assert torch.allclose(bert_cls, cross_encoder_cls, atol=5e-6)
                bert_query = bert_out.hidden_states[layer_idx][
                    batch_idx, 1 : query_len + 1
                ]
                cross_encoder_query = cross_encoder_out.hidden_states[layer_idx].query[
                    batch_idx, 0, :query_len
                ]
                assert torch.allclose(bert_query, cross_encoder_query, atol=5e-6)
                bert_doc = bert_out.hidden_states[layer_idx][
                    batch_idx, 1 + query_len : 1 + query_len + doc_len
                ]
                cross_encoder_doc = cross_encoder_out.hidden_states[layer_idx].doc[
                    batch_idx, 0, :doc_len
                ]
                assert torch.allclose(bert_doc, cross_encoder_doc, atol=5e-6)

        bert_logits = bert_out.logits[:, 0]
        cross_encoder_logits = cross_encoder_out.logits[0]

        assert torch.allclose(bert_logits, cross_encoder_logits, atol=5e-7)

    def test_sparse_cross_encoder(
        self,
        bert: BertForSequenceClassification,
        sparse_cross_encoder: SparseCrossEncoderModelForSequenceClassification,
        tokenizer: BertTokenizerFast,
    ):
        query = "How many people live in Berlin?"
        doc = (
            "Berlin has a population of 3,520,031 registered inhabitants "
            "in an area of 891.82 square kilometers."
        )
        sparse_cross_encoder_features = tokenizer([query], [doc], return_tensors="pt")
        query_input_ids, doc_input_ids = tokenizer([query, doc]).input_ids
        query_input_ids = [torch.tensor(query_input_ids[1:])]
        doc_input_ids = [[torch.tensor(doc_input_ids[1:])]]

        with torch.no_grad():
            bert_out = bert(**sparse_cross_encoder_features)
            sparse_cross_encoder_out = sparse_cross_encoder(
                query_input_ids, doc_input_ids
            )

        bert_logits = bert_out.logits[:, 0]
        sparse_cross_encoder_logits = sparse_cross_encoder_out.logits[0]

        assert not torch.allclose(bert_logits, sparse_cross_encoder_logits, atol=5e-7)
