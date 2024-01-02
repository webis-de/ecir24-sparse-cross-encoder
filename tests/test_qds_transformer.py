from sparse_cross_encoder.model.sparse_cross_encoder import SparseCrossEncoderConfig
from qds_transformer.qds_transformer import (
    QDSTransformerDataModule,
    QDSTransformerModule,
)


def test_qds_transformer(model_name: str):
    datamodule = QDSTransformerDataModule(
        model_name,
        ir_dataset_paths=["msmarco-passage/train/triples-small"],
        max_length=32,
        batch_size=2,
    )
    datamodule.setup()
    dataloader = datamodule.train_dataloader()

    config = SparseCrossEncoderConfig(num_hidden_layers=1)
    module = QDSTransformerModule(model_name, config)

    batch = next(iter(dataloader))
    loss = module.training_step(batch, 0)
    assert loss
