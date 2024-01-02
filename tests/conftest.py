from pathlib import Path

import ir_datasets
import pytest
import transformers
from transformers import AutoTokenizer

from sparse_cross_encoder.data.datamodule import SparseCrossEncoderDataModule


@pytest.fixture(scope="session")
def config_dir() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "sparse_cross_encoder"
        / "configs"
        / "models"
    )


@pytest.fixture(scope="session")
def ir_dataset_name() -> str:
    return "msmarco-passage/train/triples-small"


@pytest.fixture(scope="session")
def model_name() -> str:
    return "cross-encoder/ms-marco-MiniLM-L-6-v2"


@pytest.fixture(scope="session")
def tokenizer(model_name: str) -> transformers.PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name)


@pytest.fixture(scope="session")
def ir_dataset(ir_dataset_name: str) -> ir_datasets.Dataset:
    return ir_datasets.load(ir_dataset_name)


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="session")
def test_run_path(test_data_dir: Path) -> Path:
    return test_data_dir / "msmarco-passage-trec-dl-2019-judged.run"


@pytest.fixture(scope="session")
def train_datamodule(model_name: str) -> SparseCrossEncoderDataModule:
    datamodule = SparseCrossEncoderDataModule(
        model_name_or_path=model_name,
        ir_dataset_paths=["msmarco-passage/train/triples-small"],
        truncate=True,
        max_length=32,
        batch_size=2,
    )
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule


@pytest.fixture(scope="session")
def predict_datamodule(
    test_run_path: Path, model_name: str
) -> SparseCrossEncoderDataModule:
    datamodule = SparseCrossEncoderDataModule(
        model_name_or_path=model_name,
        ir_dataset_paths=[test_run_path],
        truncate=True,
        max_length=32,
        batch_size=2,
    )
    datamodule.prepare_data()
    datamodule.setup(stage="predict")
    return datamodule
