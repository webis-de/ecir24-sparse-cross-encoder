from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from sparse_cross_encoder.data.datamodule import SparseCrossEncoderDataModule
from sparse_cross_encoder.model.loss_utils import MarginMSE
from sparse_cross_encoder.model.sparse_cross_encoder import SparseCrossEncoderConfig
from sparse_cross_encoder.model.sparse_cross_encoder_module import (
    SparseCrossEncoderModule,
)

config = SparseCrossEncoderConfig(
    max_position_embeddings=512,
    attention_window_size=4,
    cls_query_attention=True,
    cls_doc_attention=True,
    query_cls_attention=False,
    query_doc_attention=False,
    doc_cls_attention=True,
    doc_query_attention=True,
)
module = SparseCrossEncoderModule(
    model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2", config=config, loss_function=MarginMSE()
)

datamodule = SparseCrossEncoderDataModule(
    model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2",
    ir_dataset_path="msmarco-passage/train/kd-docpairs",
    truncate=True,
    max_length=512,
    batch_size=32,
    num_workers=4,
)

trainer = Trainer(callbacks=ModelCheckpoint(every_n_train_steps=5_000), max_steps=100_000, val_check_interval=5_000)

trainer.fit(module, datamodule)
