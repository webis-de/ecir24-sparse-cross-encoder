from typing import Any, Optional, Union

import torch
from lightning import LightningModule
from lightning.pytorch.cli import LightningCLI  # noqa: F401
from torch.optim import Optimizer

from sparse_cross_encoder.data.datamodule import SparseCrossEncoderDataModule
from sparse_cross_encoder.model.sparse_cross_encoder import (
    SparseCrossEncoderConfig,
)  # noqa: F401
from sparse_cross_encoder.model.sparse_cross_encoder_module import (
    SparseCrossEncoderModule,
)
from sparse_cross_encoder.model.warmup_schedulers import LR_SCHEDULERS  # noqa: F401
from sparse_cross_encoder.model.warmup_schedulers import (
    ConstantSchedulerWithWarmup,
    LinearSchedulerWithWarmup,
)
from sparse_cross_encoder.model import loss_utils  # noqa: F401
from sparse_cross_encoder.model.loggers import CustomWandbLogger  # noqa: F401


if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")


class SparseCrossEncoderLightningCLI(LightningCLI):
    @staticmethod
    def configure_optimizers(
        lightning_module: LightningModule,
        optimizer: Optimizer,
        lr_scheduler: Optional[
            Union[ConstantSchedulerWithWarmup, LinearSchedulerWithWarmup]
        ] = None,
    ) -> Any:
        if lr_scheduler is None:
            return optimizer

        return [optimizer], [
            {"scheduler": lr_scheduler, "interval": lr_scheduler.interval}
        ]

    def add_arguments_to_parser(self, parser):
        parser.add_lr_scheduler_args(tuple(LR_SCHEDULERS))
        parser.link_arguments("model.model_name_or_path", "data.model_name_or_path")
        parser.link_arguments(
            "trainer.max_steps", "lr_scheduler.init_args.num_training_steps"
        )


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
        SparseCrossEncoderModule,
        SparseCrossEncoderDataModule,
        save_config_kwargs={"config_filename": "pl_config.yaml", "overwrite": True},
    )


if __name__ == "__main__":
    main()
