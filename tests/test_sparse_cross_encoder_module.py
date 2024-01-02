from typing import Literal, Type
import pathlib

import pytest
import torch

from sparse_cross_encoder.data.datamodule import SparseCrossEncoderDataModule
from sparse_cross_encoder.model.sparse_cross_encoder_module import (
    SparseCrossEncoderModule,
    SparseCrossEncoderConfig,
)
from transformers import AutoConfig
from sparse_cross_encoder.model import loss_utils


class TestListwiseSparseCrossEncoderModule:
    @pytest.fixture()
    def model(
        self, model_name: str, config_dir: pathlib.Path
    ) -> SparseCrossEncoderModule:
        config_path = str(config_dir / "sparse-cross-encoder.json")
        config = SparseCrossEncoderConfig.from_pretrained(config_path)
        config.update(AutoConfig.from_pretrained(model_name).to_dict())
        config.update({"num_hidden_layers": 1})
        sparse_cross_encoder_module = SparseCrossEncoderModule(model_name, config)
        return sparse_cross_encoder_module

    def test_init(
        self,
        model: SparseCrossEncoderModule,
    ):
        assert model

    @pytest.mark.parametrize(
        "loss_function_cls",
        [
            loss_utils.RankNet,
            loss_utils.ApproxNDCG,
            loss_utils.ApproxMRR,
            loss_utils.ListNet,
            loss_utils.ListMLE,
            loss_utils.LambdaNDCG1,
            loss_utils.LambdaNDCG2,
            loss_utils.LambdaARP1,
            loss_utils.LambdaARP2,
            loss_utils.NeuralNDCG,
            loss_utils.NeuralMRR,
            loss_utils.LocalizedContrastive,
        ],
    )
    def test_training_step(
        self,
        model: SparseCrossEncoderModule,
        train_datamodule: SparseCrossEncoderDataModule,
        loss_function_cls: Type[loss_utils.LossFunc],
    ):
        dataloader = train_datamodule.train_dataloader()
        batch = next(iter(dataloader))
        model.loss_function = loss_function_cls()
        loss = model.training_step(batch, 0)
        assert loss.requires_grad
        assert loss > 0
        assert not torch.isnan(loss)

    def test_prediction_step(
        self,
        model: SparseCrossEncoderModule,
        predict_datamodule: SparseCrossEncoderDataModule,
    ):
        dataloader = predict_datamodule.predict_dataloader()[0]
        batch = next(iter(dataloader))
        with torch.inference_mode():
            logits = model.predict_step(batch, 0)
            assert logits[0].shape[0] == 100
