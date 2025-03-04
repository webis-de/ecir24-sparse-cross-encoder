from lightning import Trainer

from sparse_cross_encoder.data.datamodule import SparseCrossEncoderDataModule
from sparse_cross_encoder.model.callbacks import PredictionWriter
from sparse_cross_encoder.model.sparse_cross_encoder_module import (
    SparseCrossEncoderModule,
)

module = SparseCrossEncoderModule(model_name_or_path="webis/sparse-cross-encoder-4-512")
data_module = SparseCrossEncoderDataModule(
    model_name_or_path="webis/sparse-cross-encoder-4-512",
    ir_dataset_path="msmarco-passage/trec-dl-2019/judged",
    max_length=512,
    batch_size=1,
    depth=1000,
)
trainer = Trainer(logger=False, callbacks=[PredictionWriter(output_path="run.txt")])

trainer.predict(module, datamodule=data_module, return_predictions=False)
