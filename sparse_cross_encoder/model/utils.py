from pathlib import Path

import torch
from sparse_cross_encoder.model.sparse_cross_encoder import (
    SparseCrossEncoderModelForSequenceClassification,
)


def extend_position_embeddings(run_path: Path, max_position_embeddings: int) -> None:
    # NOTE DON'T PASS ALREADY CONVERTED MODEL!!!
    model = SparseCrossEncoderModelForSequenceClassification.from_pretrained(
        run_path / "files" / "huggingface_checkpoint",
        max_position_embeddings=max_position_embeddings,
    )
    ckpt_paths = list((run_path / "files" / "checkpoints").glob("epoch=*.ckpt"))
    assert len(ckpt_paths) == 1
    ckpt_path = ckpt_paths[0]
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    extended_state_dict = model.state_dict()
    new_state_dict = checkpoint["state_dict"].copy()
    for key in extended_state_dict.keys():
        if (
            extended_state_dict[key].shape
            != new_state_dict["sparse_cross_encoder." + key].shape
        ):
            new_state_dict["sparse_cross_encoder." + key] = extended_state_dict[key]
        else:
            assert torch.allclose(
                extended_state_dict[key], new_state_dict["sparse_cross_encoder." + key]
            )
    checkpoint["state_dict"] = new_state_dict
    torch.save(checkpoint, ckpt_path)
    model.save_pretrained(run_path / "files" / "huggingface_checkpoint")


def reduce_position_embeddings(run_path: Path, max_position_embeddings: int) -> None:
    try:
        model = SparseCrossEncoderModelForSequenceClassification.from_pretrained(
            run_path / "files" / "huggingface_checkpoint",
            max_position_embeddings=max_position_embeddings,
        )
    except:
        # only temporary. remove after current qds transformer training has finished
        pytorch_model_bin = (
            run_path / "files" / "huggingface_checkpoint" / "pytorch_model.bin"
        )
        state_dict = torch.load(pytorch_model_bin, map_location="cpu")
        state_dict["bert.embeddings.position_embeddings.weight"] = state_dict[
            "bert.embeddings.position_embeddings.weight"
        ][:max_position_embeddings]
        state_dict["bert.embeddings.position_ids"] = state_dict[
            "bert.embeddings.position_ids"
        ][:, :max_position_embeddings]
        torch.save(state_dict, pytorch_model_bin)
        model = SparseCrossEncoderModelForSequenceClassification.from_pretrained(
            run_path / "files" / "huggingface_checkpoint",
            max_position_embeddings=max_position_embeddings,
        )
    model.config.base_max_position_embeddings = max_position_embeddings
    ckpt_paths = list((run_path / "files" / "checkpoints").glob("epoch=*.ckpt"))
    assert len(ckpt_paths) == 1
    ckpt_path = ckpt_paths[0]
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    state_dict = checkpoint["state_dict"]
    state_dict[
        "sparse_cross_encoder.bert.embeddings.position_embeddings.weight"
    ] = state_dict["sparse_cross_encoder.bert.embeddings.position_embeddings.weight"][
        :max_position_embeddings
    ]
    state_dict["sparse_cross_encoder.bert.embeddings.position_ids"] = state_dict[
        "sparse_cross_encoder.bert.embeddings.position_ids"
    ][:, :max_position_embeddings]
    torch.save(checkpoint, ckpt_path)

    model.bert.embeddings.position_embeddings.num_embeddings = max_position_embeddings
    model.bert.embeddings.position_embeddings.weight = torch.nn.Parameter(
        model.bert.embeddings.position_embeddings.weight[:max_position_embeddings]
    )
    model.bert.embeddings.position_ids = model.bert.embeddings.position_ids[
        :, :max_position_embeddings
    ]
    model.save_pretrained(run_path / "files" / "huggingface_checkpoint")
