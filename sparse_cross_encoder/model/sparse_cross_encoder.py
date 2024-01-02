import math
import os
import re
from dataclasses import dataclass, is_dataclass
from functools import reduce
from copy import deepcopy
from typing import Any, Dict, Sequence, Literal, Optional, Tuple, List

import torch
import torch.utils.checkpoint
from sklearn import linear_model
from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from transformers.utils.generic import ModelOutput
from window_matmul import unwindow_matmul, window_matmul

logger = logging.get_logger(__name__)

ALLOW_CAT_FORWARD = True


class SparseCrossEncoderConfig(PretrainedConfig):
    model_type: str = "bert"

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        position_embedding_type: Literal["absolute", "relative"] = "absolute",
        use_cache: bool = True,
        token_type_embedding: bool = True,
        pad_token_id: Optional[int] = None,
        cls_token_id: Optional[int] = None,
        depth: int = 100,
        attention_window_size: Optional[int] = None,
        cls_query_attention: bool = True,
        cls_doc_attention: bool = True,
        query_cls_attention: bool = True,
        query_doc_attention: bool = True,
        doc_cls_attention: bool = True,
        doc_query_attention: bool = True,
        **kwargs,
    ):
        # num_labels needs to be set, otherwise transformers will complain
        super().__init__(num_labels=1, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.token_type_embedding = token_type_embedding
        self.depth = depth
        self.attention_window_size = attention_window_size
        self.cls_query_attention = cls_query_attention
        self.cls_doc_attention = cls_doc_attention
        self.query_cls_attention = query_cls_attention
        self.query_doc_attention = query_doc_attention
        self.doc_cls_attention = doc_cls_attention
        self.doc_query_attention = doc_query_attention
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id

    def get_diff_dict(self) -> Dict[str, Any]:
        diff_dict = self.to_diff_dict()
        default_dict = SparseCrossEncoderConfig().to_diff_dict()
        keys = list(diff_dict.keys())
        for key in keys:
            try:
                if diff_dict[key] == default_dict[key]:
                    del diff_dict[key]
            except KeyError:
                pass
        return diff_dict


def nested_dataclass(*args, **kwargs):
    def wrapper(cls):
        cls = dataclass(cls, **kwargs)
        original_init = cls.__init__

        def __init__(self, *args, **kwargs):
            for name, value in kwargs.items():
                field_type = cls.__annotations__.get(name, None)
                if is_dataclass(field_type) and isinstance(value, dict):
                    new_obj = field_type(**value)
                    kwargs[name] = new_obj
            original_init(self, *args, **kwargs)

        cls.__init__ = __init__
        return cls

    return wrapper(args[0]) if args else wrapper


@dataclass
class HiddenStates:
    cls: torch.Tensor
    query: torch.Tensor
    doc: torch.Tensor

    def __repr__(self) -> str:
        return f"<HiddenStates {id(self)}>"


@dataclass
class Vectors:
    query_layer: torch.Tensor
    key_layer: torch.Tensor
    value_layer: torch.Tensor

    def __repr__(self) -> str:
        return f"<Vectors {id(self)}>"


@dataclass
@nested_dataclass
class HiddenVectors:
    cls: Vectors
    query: Vectors
    doc: Vectors

    def __repr__(self) -> str:
        return f"<HiddenVectors {id(self)}>"


@dataclass
class AttentionMasks:
    cls: Optional[torch.Tensor]
    query: Optional[torch.Tensor]
    doc: Optional[torch.Tensor]

    def __repr__(self) -> str:
        return f"<AttentionMasks {id(self)}>"


@dataclass
class AttentionInput:
    query_layer: torch.Tensor
    key_layers: List[torch.Tensor]
    value_layers: List[torch.Tensor]
    attention_masks: List[Optional[torch.Tensor]]

    def __repr__(self) -> str:
        return f"<AttentionInput {id(self)}>"


@dataclass
@nested_dataclass
class SelfAttentionInput:
    hidden_vectors: HiddenVectors
    attention_masks: AttentionMasks

    def __repr__(self) -> str:
        return f"<SelfAttentionInput {id(self)}>"


@dataclass
@nested_dataclass
class EncoderInput:
    hidden_states: HiddenStates
    attention_masks: AttentionMasks

    def __repr__(self) -> str:
        return f"<EncoderInput {id(self)}>"


def list_softmax(tensors: Sequence[torch.Tensor], dim: int = -1) -> List[torch.Tensor]:
    # manual softmax, subtract max for numerical stability
    # https://stackoverflow.com/questions/42599498/numerically-stable-softmax
    max_tensors = [
        attention_score.max(dim=dim, keepdim=True)[0]
        for attention_score in tensors
        if attention_score.numel()
    ]
    if not max_tensors:
        return tensors
    max_attention_score = reduce(torch.max, max_tensors)
    numerators = [
        torch.exp(attention_score - max_attention_score) for attention_score in tensors
    ]
    denominator = reduce(
        torch.add, (numerator.sum(dim, keepdim=True) for numerator in numerators)
    )
    return [numerator / denominator for numerator in numerators]


def to_windowed_attention_mask(
    attention_mask: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    full_window_size = window_size * 2 + 1
    attention_mask = attention_mask.squeeze(-2)
    *sizes, seq_len = attention_mask.size()
    attention_mask = torch.nn.functional.pad(
        attention_mask,
        (window_size, window_size),
        value=torch.finfo(attention_mask.dtype).min,
    )
    new_shape = (*sizes, seq_len, full_window_size)
    new_stride = list(attention_mask.stride())
    new_stride.insert(-1, 1)
    new_stride = tuple(new_stride)
    attention_mask = attention_mask.as_strided(new_shape, new_stride)
    return attention_mask


def transpose_for_scores(
    x: torch.Tensor, num_attention_heads: int, attention_head_size: int
) -> torch.Tensor:
    while x.ndim < 3:
        x = x.unsqueeze(0)
    new_x_shape = x.size()[:-1] + (
        num_attention_heads,
        attention_head_size,
    )
    x = x.view(new_x_shape)
    permute = list(range(x.ndim))
    permute[-4], permute[-3], permute[-2] = permute[-2], permute[-4], permute[-3]
    x = x.permute(*permute)
    return x


def untranspose_for_scores(x: torch.Tensor, all_head_size: int) -> torch.Tensor:
    x = x.permute(0, 2, 3, 1, 4)
    shapes = x.shape[:-2]
    x = x.reshape(*shapes, all_head_size)
    return x


@dataclass
class SparseCrossEncoderModelOutput(ModelOutput):
    logits: List[torch.Tensor] = None
    last_hidden_state: HiddenStates = None
    pooler_output: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[HiddenStates]] = None

    def __repr__(self) -> str:
        return f"<SparseCrossEncoderModelOutput {id(self)}>"


class SparseCrossEncoderEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: SparseCrossEncoderConfig):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
        )

    def _find_trained_embedding_size(self):
        embeddings = self.position_embeddings.weight.data

        def to_windowed(embeddings: torch.Tensor, window: int) -> torch.Tensor:
            seq_len, hidden_dim = embeddings.size()
            num_windows = max(seq_len, window) - window + 1
            new_stride = tuple(list(embeddings.stride()[:-1]) + [hidden_dim, 1])
            new_shape = (num_windows, window, hidden_dim)
            return embeddings.as_strided(new_shape, new_stride)

        def create_regression_data(
            embeddings: torch.Tensor, window: int
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            X = to_windowed(embeddings, window)[:-1]
            X = X.reshape(X.shape[0], -1)
            Y = embeddings[window:]
            return X, Y

        def fit_regression(embeddings: torch.Tensor, window: int) -> linear_model.Ridge:
            X, Y = create_regression_data(embeddings, window)
            reg = linear_model.Ridge()
            reg.fit(X, Y)
            return reg

        # find the optimal sequence length cut-off point for training the regression
        # since some bert models haven't trained the longer sequence lengths extensively
        # it is better to cut off the sequence length (starting backwards) at the point
        # where the regression model is no longer improves
        window = 5
        patience = 3
        step = 5
        train_idcs = list(range(window + 1, embeddings.shape[0] - window - 1, step))
        train_idcs = train_idcs[::-1]
        prev_loss = float("inf")
        repeated_greater = 0
        train_idx = None
        losses = []
        for train_idx in train_idcs:
            train_encodings = embeddings[:train_idx]
            test_encodings = embeddings[train_idx:]
            reg = fit_regression(train_encodings, window)
            X, Y = create_regression_data(test_encodings, window)
            loss = torch.nn.functional.mse_loss(torch.tensor(reg.predict(X)), Y)
            losses.append(loss)
            if loss > prev_loss:
                repeated_greater += 1
            else:
                repeated_greater = 0
            if repeated_greater >= patience:
                break
            prev_loss = loss
        assert train_idx is not None
        return train_idx

    def extend_position_embeddings(self, max_position_embeddings: int) -> None:
        embeddings = self.position_embeddings.weight.data
        trained_embedding_idx = self._find_trained_embedding_size()
        embeddings = embeddings[:trained_embedding_idx]
        indices = torch.linspace(
            0,
            embeddings.shape[0] - 1,
            max_position_embeddings,
        )
        start = indices.floor().long()
        end = indices.ceil().long()
        weight = indices - start.float()
        extended_embeddings = torch.lerp(
            embeddings[start], embeddings[end], weight[:, None]
        )
        self.position_embeddings.num_embeddings = max_position_embeddings
        self.position_embeddings.weight.data = extended_embeddings
        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1)),
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        offset: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
        token_type_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if seq_len is None:
            seq_len = input_ids.shape[-1]

        input_embeddings = self.word_embeddings(input_ids)
        if not self.config.token_type_embedding or token_type_id is None:
            token_type_embeddings = self.token_type_embeddings(
                torch.zeros(1, dtype=torch.long, device=self.position_ids.device)
            )
        else:
            token_type_embeddings = self.token_type_embeddings(
                torch.full(
                    (1,),
                    token_type_id,
                    dtype=torch.long,
                    device=self.position_ids.device,
                )
            )
        embeddings = input_embeddings + token_type_embeddings

        if self.position_embedding_type == "absolute":
            position_ids = self.position_ids[:, :seq_len]
            # add num_docs dim
            position_ids = position_ids.unsqueeze(1)
            if offset is not None:
                position_ids = position_ids + offset
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SparseCrossEncoderSelfAttention(nn.Module):
    def __init__(self, config: SparseCrossEncoderConfig, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the "
                f"number of attention heads ({config.num_attention_heads})"
            )
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )

    def forward(self, enc_inp: EncoderInput) -> HiddenStates:
        hidden_vectors = []
        for name in ("cls", "query", "doc"):
            hidden_state = getattr(enc_inp.hidden_states, name)
            vectors = Vectors(
                *(
                    transpose_for_scores(
                        layer(hidden_state),
                        self.num_attention_heads,
                        self.attention_head_size,
                    )
                    for layer in [self.query, self.key, self.value]
                )
            )
            hidden_vectors.append(vectors)

        inp = SelfAttentionInput(
            HiddenVectors(*hidden_vectors),
            enc_inp.attention_masks,
        )
        cls = self.cls_forward(inp)
        query = self.query_forward(inp)
        doc = self.doc_forward(inp)
        hidden_states = HiddenStates(cls, query, doc)
        return hidden_states

    def _forward(
        self,
        inp: AttentionInput,
        attention_window_sizes: Tuple[Optional[int], ...],
    ) -> torch.Tensor:
        if not any(attention_window_sizes) and ALLOW_CAT_FORWARD:
            # no windowed attention, we can simply concatenate the key and value layers
            context_layer = self._cat_forward(inp)
        else:
            context_layer = self._window_forward(inp, attention_window_sizes)
        context_layer = untranspose_for_scores(context_layer, self.all_head_size)
        return context_layer

    def _cat_forward(self, inp: AttentionInput) -> torch.Tensor:
        num_docs = max(key_layer.shape[2] for key_layer in inp.key_layers)
        key_layers = [
            key_layer.expand(-1, -1, num_docs, -1, -1) for key_layer in inp.key_layers
        ]
        value_layers = [
            value_layer.expand(-1, -1, num_docs, -1, -1)
            for value_layer in inp.value_layers
        ]
        key_layer = torch.cat(key_layers, dim=3)
        value_layer = torch.cat(value_layers, dim=3)
        attention_masks = [
            attention_mask.expand(-1, -1, num_docs, -1, -1)
            for attention_mask in inp.attention_masks
            if attention_mask is not None
        ]
        attention_mask = None
        if attention_masks:
            attention_mask = torch.cat(attention_masks, dim=4)
        context_layer = torch.nn.functional.scaled_dot_product_attention(
            inp.query_layer,
            key_layer,
            value_layer,
            attention_mask,
            self.dropout.p if self.training else 0,
        )
        return context_layer

    def _window_forward(
        self,
        inp: AttentionInput,
        attention_window_sizes: Tuple[Optional[int], ...],
    ) -> torch.Tensor:
        # TODO use nested tensors when broadcasting is supported
        # https://pytorch.org/docs/stable/nested.html
        query_layer = inp.query_layer
        key_layers = inp.key_layers
        value_layers = inp.value_layers
        attention_masks = inp.attention_masks

        # Take the dot product between "query" and "key" to get the raw attention scores
        attention_scores = []
        for key_layer, window_size in zip(key_layers, attention_window_sizes):
            if window_size is not None:
                attention_scores.append(
                    window_matmul(query_layer, key_layer.transpose(-1, -2), window_size)
                )
            else:
                attention_scores.append(
                    torch.matmul(query_layer, key_layer.transpose(-1, -2))
                )

        attention_scores = [
            attention_score / math.sqrt(self.attention_head_size)
            for attention_score in attention_scores
        ]

        iterator = enumerate(
            zip(attention_masks, attention_scores, attention_window_sizes)
        )
        for idx, (attention_mask, attention_score, attention_window_size) in iterator:
            if attention_mask is not None:
                if attention_window_size is not None:
                    attention_mask = to_windowed_attention_mask(
                        attention_mask, attention_window_size
                    )
                attention_score = attention_score + attention_mask
                attention_score = attention_score.clamp(
                    torch.finfo(attention_score.dtype).min
                )
                attention_scores[idx] = attention_score

        # Normalize the attention scores to probabilities
        attention_probs = list_softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = [
            self.dropout(attention_prob) for attention_prob in attention_probs
        ]

        context_layers = []
        iterator = zip(attention_probs, value_layers, attention_window_sizes)
        for attention_prob, value_layer, window_size in iterator:
            attention_prob = attention_prob.to(value_layer)
            if window_size is not None:
                context_layers.append(
                    unwindow_matmul(attention_prob, value_layer, window_size)
                )
            else:
                context_layers.append(torch.matmul(attention_prob, value_layer))
        context_layer = reduce(torch.add, context_layers)
        return context_layer

    def cls_forward(self, inp: SelfAttentionInput) -> torch.Tensor:
        query_layer = inp.hidden_vectors.cls.query_layer
        key_layers = [inp.hidden_vectors.cls.key_layer]
        value_layers = [inp.hidden_vectors.cls.value_layer]
        attention_masks = [inp.attention_masks.cls]

        if self.config.cls_query_attention:
            key_layers.append(inp.hidden_vectors.query.key_layer)
            value_layers.append(inp.hidden_vectors.query.value_layer)
            attention_masks.append(inp.attention_masks.query)
        if self.config.cls_doc_attention:
            key_layers.append(inp.hidden_vectors.doc.key_layer)
            value_layers.append(inp.hidden_vectors.doc.value_layer)
            attention_masks.append(inp.attention_masks.doc)

        att_inp = AttentionInput(
            query_layer,
            key_layers,
            value_layers,
            attention_masks,
        )
        attention_window_sizes = (None,) * len(key_layers)
        context_layer = self._forward(att_inp, attention_window_sizes)
        return context_layer

    def query_forward(self, inp: SelfAttentionInput) -> torch.Tensor:
        query_layer = inp.hidden_vectors.query.query_layer
        key_layers = [inp.hidden_vectors.query.key_layer]
        value_layers = [inp.hidden_vectors.query.value_layer]
        attention_masks = [inp.attention_masks.query]

        if self.config.query_cls_attention:
            key_layers.append(inp.hidden_vectors.cls.key_layer)
            value_layers.append(inp.hidden_vectors.cls.value_layer)
            attention_masks.append(inp.attention_masks.cls)
        if self.config.query_doc_attention:
            key_layers.append(inp.hidden_vectors.doc.key_layer)
            value_layers.append(inp.hidden_vectors.doc.value_layer)
            attention_masks.append(inp.attention_masks.doc)

        att_inp = AttentionInput(
            query_layer,
            key_layers,
            value_layers,
            attention_masks,
        )
        attention_window_sizes = (None,) * len(key_layers)
        context_layer = self._forward(att_inp, attention_window_sizes)
        return context_layer

    def doc_forward(self, inp: SelfAttentionInput) -> torch.Tensor:
        query_layer = inp.hidden_vectors.doc.query_layer
        key_layers = [inp.hidden_vectors.doc.key_layer]
        value_layers = [inp.hidden_vectors.doc.value_layer]
        attention_masks = [inp.attention_masks.doc]
        attention_window_sizes = [self.config.attention_window_size]

        if self.config.doc_cls_attention:
            key_layers.append(inp.hidden_vectors.cls.key_layer)
            value_layers.append(inp.hidden_vectors.cls.value_layer)
            attention_masks.append(inp.attention_masks.cls)
            attention_window_sizes.append(None)
        if self.config.doc_query_attention:
            key_layers.append(inp.hidden_vectors.query.key_layer)
            value_layers.append(inp.hidden_vectors.query.value_layer)
            attention_masks.append(inp.attention_masks.query)
            attention_window_sizes.append(None)

        att_inp = AttentionInput(
            query_layer,
            key_layers,
            value_layers,
            attention_masks,
        )
        attention_window_sizes = tuple(attention_window_sizes)
        context_layer = self._forward(att_inp, attention_window_sizes)
        return context_layer


class SparseCrossEncoderSelfOutput(nn.Module):
    def __init__(self, config: SparseCrossEncoderConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: HiddenStates, input_tensor: HiddenStates
    ) -> HiddenStates:
        cls = self._forward(hidden_states.cls, input_tensor.cls)
        query = self._forward(hidden_states.query, input_tensor.query)
        doc = self._forward(hidden_states.doc, input_tensor.doc)
        hidden_states = HiddenStates(cls, query, doc)
        return hidden_states

    def _forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SparseCrossEncoderAttention(nn.Module):
    def __init__(self, config: SparseCrossEncoderConfig, position_embedding_type=None):
        super().__init__()
        self.self = SparseCrossEncoderSelfAttention(
            config, position_embedding_type=position_embedding_type
        )
        self.output = SparseCrossEncoderSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, inp: EncoderInput) -> HiddenStates:
        hidden_states = self.self(inp)
        hidden_states = self.output(hidden_states, inp.hidden_states)
        return hidden_states


class SparseCrossEncoderIntermediate(nn.Module):
    def __init__(self, config: SparseCrossEncoderConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class SparseCrossEncoderOutput(nn.Module):
    def __init__(self, config: SparseCrossEncoderConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SparseCrossEncoderLayer(nn.Module):
    def __init__(self, config: SparseCrossEncoderConfig):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = SparseCrossEncoderAttention(config)
        self.intermediate = SparseCrossEncoderIntermediate(config)
        self.output = SparseCrossEncoderOutput(config)

    def forward(self, inp: EncoderInput) -> HiddenStates:
        hidden_states = self.attention(inp)

        cls = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            hidden_states.cls,
        )
        query = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            hidden_states.query,
        )
        doc = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            hidden_states.doc,
        )

        return HiddenStates(cls, query, doc)

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class SparseCrossEncoderEncoder(nn.Module):
    def __init__(self, config: SparseCrossEncoderConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [SparseCrossEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        inp: EncoderInput,
        output_hidden_states: bool = False,
    ) -> SparseCrossEncoderModelOutput:
        all_hidden_states = () if output_hidden_states else None
        hidden_states = inp.hidden_states

        for layer_idx, layer_module in enumerate(self.layer):
            if all_hidden_states is not None:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states = layer_module(inp)
            inp.hidden_states = hidden_states

        if all_hidden_states is not None:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return SparseCrossEncoderModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class SparseCrossEncoderPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SparseCrossEncoderConfig
    base_model_prefix = "bert"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike | None,
        *model_args,
        config: PretrainedConfig | str | os.PathLike | None = None,
        **kwargs,
    ) -> "SparseCrossEncoderPreTrainedModel":
        try:
            model = super().from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                **kwargs,
            )
        except RuntimeError as exc:
            if "embeddings.position_embeddings.weight" not in str(exc):
                raise exc
            if isinstance(config, PretrainedConfig):
                checkpoint_config = deepcopy(config)
            else:
                config_path = (
                    config if config is not None else pretrained_model_name_or_path
                )
                assert config_path is not None
                checkpoint_config, kwargs = cls.config_class.from_pretrained(
                    config_path,
                    return_unused_kwargs=True,
                    **kwargs,
                )
            pattern = r"torch.Size\(\[(\d+), {hidden_size}\]\)".format(
                hidden_size=checkpoint_config.hidden_size
            )
            sizes = re.findall(pattern, str(exc))
            assert len(sizes) == 2
            checkpoint_size, model_size = map(int, sizes)
            checkpoint_config.max_position_embeddings = checkpoint_size
            model = super().from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=checkpoint_config,
                **kwargs,
            )
            setattr(model.config, "max_position_embeddings", model_size)
            for module in model.modules():
                if isinstance(module, SparseCrossEncoderEmbeddings):
                    module.extend_position_embeddings(model_size)
                    assert module.position_embeddings.num_embeddings == model_size

        return model


class SparseCrossEncoderModelForSequenceClassification(
    SparseCrossEncoderPreTrainedModel
):
    def __init__(
        self, config: SparseCrossEncoderConfig, model_type: Literal["bert", "electra"]
    ):
        super().__init__(config)
        self.config = config
        self.model_type = model_type
        self.add_module(model_type, SparseCrossEncoderModel(config))

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        query_input_ids: Sequence[torch.Tensor],
        doc_input_ids: Sequence[Sequence[torch.Tensor]],
        output_hidden_states: bool = False,
    ) -> SparseCrossEncoderModelOutput:
        output: SparseCrossEncoderModelOutput = getattr(self, self.model_type).forward(
            query_input_ids, doc_input_ids, output_hidden_states
        )

        if output.pooler_output is None:
            cls_output = output.last_hidden_state.cls
        else:
            cls_output = output.pooler_output
        logits = self.classifier(cls_output)[..., 0, 0]
        output_logits = [
            logits[batch_idx, : len(doc_input_ids[batch_idx])]
            for batch_idx in range(logits.shape[0])
        ]

        output.logits = output_logits

        return output


class SparseCrossEncoderPooler(nn.Module):
    def __init__(self, config: SparseCrossEncoderConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SparseCrossEncoderModel(SparseCrossEncoderPreTrainedModel):
    def __init__(
        self, config: SparseCrossEncoderConfig, add_pooling_layer: bool = True
    ):
        super().__init__(config)
        self.config = config
        self.embeddings = SparseCrossEncoderEmbeddings(config)
        self.encoder = SparseCrossEncoderEncoder(config)

        self.pooler = SparseCrossEncoderPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """input should be attention mask where 1 = attention to and 0 = no attention

        Args:
            attention_mask (torch.Tensor): batch_size x num_docs x seq_len

        Returns:
            torch.Tensor: batch_size x 1 x 1 x num_docs * seq_len
            batch_size x head_dim x from_seq_len x num_docs * seq_len
        """
        batch_size, num_docs, seq_len = attention_mask.shape
        attention_mask = attention_mask.view(batch_size, 1, num_docs, 1, seq_len)

        attention_mask = (1 - attention_mask) * torch.finfo(self.dtype).min

        return attention_mask

    def get_attention_masks(self, input_tensors: HiddenStates) -> AttentionMasks:
        query = torch.where(
            input_tensors.query == self.config.pad_token_id,
            torch.zeros_like(input_tensors.query),
            torch.ones_like(input_tensors.query),
        )
        doc = torch.where(
            input_tensors.doc == self.config.pad_token_id,
            torch.zeros_like(input_tensors.doc),
            torch.ones_like(input_tensors.doc),
        )
        cls = doc.any(-1).unsqueeze(-1).to(doc.dtype)
        cls = self.get_extended_attention_mask(cls)
        query = self.get_extended_attention_mask(query)
        doc = self.get_extended_attention_mask(doc)
        attention_masks = AttentionMasks(cls, query, doc)
        return attention_masks

    def get_hidden_states(self, input_tensors: HiddenStates) -> HiddenStates:
        cls = self.embeddings(input_tensors.cls)
        query_offsets = 1
        query_length = (input_tensors.query != self.config.pad_token_id).sum(-1)
        doc_offsets = query_length.unsqueeze(-1) + 1
        query = self.embeddings(input_tensors.query, offset=query_offsets)
        doc = self.embeddings(input_tensors.doc, offset=doc_offsets, token_type_id=1)
        return HiddenStates(cls, query, doc)

    def get_input_tensors(
        self,
        query_input_ids: Sequence[torch.Tensor],
        doc_input_ids: Sequence[Sequence[torch.Tensor]],
    ) -> HiddenStates:
        max_doc_len = max(
            input_ids.shape[0]
            for batch_input_ids in doc_input_ids
            for input_ids in batch_input_ids
        )
        max_num_docs = max(len(batch_input_ids) for batch_input_ids in doc_input_ids)

        query = torch.nn.utils.rnn.pad_sequence(
            query_input_ids,
            batch_first=True,
            padding_value=self.config.pad_token_id
            if self.config.pad_token_id is not None
            else 0,
        )
        query = query.unsqueeze(1)

        for key, value in self.config.to_dict().items():
            if key.startswith("query_") and key.endswith("_attention") and value:
                query = query.expand(-1, max_num_docs, -1)
                break

        padded_cls_input_ids = []
        padded_doc_input_ids = []
        for batch_input_ids in doc_input_ids:
            padded_input_ids = []
            cls_tokens = [self.config.cls_token_id]
            batch_cls_input_ids = [cls_tokens] * len(batch_input_ids)
            batch_cls_input_ids += [[self.config.pad_token_id] * len(cls_tokens)] * (
                max_num_docs - len(batch_cls_input_ids)
            )
            padded_cls_input_ids.append(
                torch.tensor(batch_cls_input_ids, device=self.device)
            )
            for input_ids in batch_input_ids:
                padded_input_ids.append(
                    torch.nn.functional.pad(
                        input_ids,
                        (0, max_doc_len - input_ids.shape[-1]),
                        value=self.config.pad_token_id,
                    )
                )
            dummy = torch.full(
                (max_doc_len,),
                self.config.pad_token_id if self.config.pad_token_id is not None else 0,
                device=self.device,
                dtype=torch.long,
            )
            padded_input_ids += [dummy] * (max_num_docs - len(padded_input_ids))
            padded_doc_input_ids.append(torch.stack(padded_input_ids))
        cls = torch.stack(padded_cls_input_ids)
        doc = torch.stack(padded_doc_input_ids)

        input_tensors = HiddenStates(cls, query, doc)

        return input_tensors

    def preprocess(
        self,
        query_input_ids: Sequence[torch.Tensor],
        doc_input_ids: Sequence[Sequence[torch.Tensor]],
    ) -> EncoderInput:
        input_tensors = self.get_input_tensors(query_input_ids, doc_input_ids)
        hidden_states = self.get_hidden_states(input_tensors)
        attention_masks = self.get_attention_masks(input_tensors)
        inp = EncoderInput(hidden_states, attention_masks)
        return inp

    def encode(
        self, inp: EncoderInput, output_hidden_states: bool = False
    ) -> SparseCrossEncoderModelOutput:
        output: SparseCrossEncoderModelOutput = self.encoder(inp, output_hidden_states)

        output.pooler_output = (
            self.pooler(output.last_hidden_state.cls)
            if self.pooler is not None
            else None
        )

        return output

    def forward(
        self,
        query_input_ids: Sequence[torch.Tensor],
        doc_input_ids: Sequence[Sequence[torch.Tensor]],
        output_hidden_states: bool = False,
    ) -> SparseCrossEncoderModelOutput:
        inp = self.preprocess(query_input_ids, doc_input_ids)
        output = self.encode(inp, output_hidden_states)
        return output
