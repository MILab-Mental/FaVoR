import random
import tempfile
from pathlib import Path

import pytest
import torch

from app.main import load_config
from app.pretrain_audio_lejepa.train import load_checkpoint, save_checkpoint
from datasets.audio_lejepa.augment import augment_local_waveform
from datasets.audio_lejepa import collate_audio_lejepa, make_audio_views
from models.audio_lejepa import AudioLeEncoder, build_audio_lejepa, load_emotion2vec_encoder
from models.video_lejepa import LeJEPALoss, ModelEMA


def tiny_cfg(pooling="extra_token"):
    return {
        "audio": {"sample_rate": 100, "process_seconds": 2.56, "max_process_seconds": 2.56},
        "backbone": {
            "mode": "emotion2vec", "conv_pos_depth": 1, "conv_pos_width": 3,
            "conv_pos_groups": 2, "prenet_depth": 1, "num_extra_tokens": 2,
        },
        "extractor": {"conv_layers": [[8, 4, 2], [8, 2, 2]], "mode": "layer_norm"},
        "encoder": {
            "depth": 1, "d_model": 8, "nhead": 2, "dim_feedforward": 16,
            "dropout": 0.0, "attention_dropout": 0.0, "qkv_bias": True,
            "layer_norm_first": False, "norm_eps": 1e-6, "ffn_targets": True,
        },
        "pooling": {"type": pooling, "index": 0},
        "projector": {"hidden_dim": 16, "output_dim": 6},
    }


def test_views_inside_anchor_and_short_audio_policies():
    random.seed(3)
    waveform = torch.arange(700, dtype=torch.float32)
    sample = make_audio_views(waveform, sample_rate=100, global_seconds=4, local_seconds=2, local_views=4)
    assert sample["global_audio"].shape == (400,)
    assert sample["local_audio"].shape == (4, 200)
    assert (sample["local_intervals"][:, 0] >= 0).all()
    assert (sample["local_intervals"][:, 1] <= 400).all()

    repeated = make_audio_views(torch.ones(40), sample_rate=100, global_seconds=1, local_seconds=.5, short_audio_policy="repeat")
    padded = make_audio_views(torch.ones(40), sample_rate=100, global_seconds=1, local_seconds=.5, short_audio_policy="zero_pad")
    assert repeated["was_repeated"] and repeated["global_length"] == 100
    assert padded["was_padded"] and padded["global_length"] == 40
    with pytest.raises(ValueError):
        make_audio_views(torch.ones(40), sample_rate=100, global_seconds=1, local_seconds=.5, short_audio_policy="skip")


def test_augmentation_fixed_seed_is_reproducible_and_bounded():
    cfg = {
        "gain": {"enabled": True, "probability": 1, "min_db": -6, "max_db": 6},
        "speed": {"enabled": True, "probability": 1, "min_rate": .95, "max_rate": 1.05},
        "polarity": {"enabled": True, "probability": 1},
        "time_mask": {"enabled": True, "probability": 1, "max_ratio": .1},
    }
    first_generator = torch.Generator().manual_seed(19)
    second_generator = torch.Generator().manual_seed(19)
    first, stats = augment_local_waveform(torch.linspace(-1, 1, 100), cfg, generator=first_generator)
    second, _ = augment_local_waveform(torch.linspace(-1, 1, 100), cfg, generator=second_generator)
    assert torch.equal(first, second)
    assert first.shape == (100,) and all(stats.values())


def test_extra_tokens_padding_mask_and_pooling_contract():
    lengths = torch.tensor([256, 128])
    waves = torch.randn(2, 256)
    extra_encoder = AudioLeEncoder(tiny_cfg("extra_token"))
    output = extra_encoder(waves, lengths, return_tokens=True)
    assert output.extra_tokens.shape == (2, 2, 8)
    assert output.tokens.shape[1] == extra_encoder.backbone.feature_extractor.output_length(256)
    assert output.padding_mask[1].any() and not output.padding_mask[0].any()
    assert torch.equal(output.embedding, output.extra_tokens[:, 0])

    mean_encoder = AudioLeEncoder(tiny_cfg("masked_mean"))
    mean_encoder.load_state_dict(extra_encoder.state_dict())
    mean_output = mean_encoder(waves, lengths, return_tokens=True)
    valid = (~mean_output.padding_mask).unsqueeze(-1)
    expected = (mean_output.tokens * valid).sum(1) / valid.sum(1)
    assert torch.allclose(mean_output.embedding, expected)


def test_shared_backbone_different_durations_and_backward():
    model = build_audio_lejepa(tiny_cfg())
    global_audio = torch.randn(2, 256)
    local_audio = torch.randn(2, 2, 128)
    embeddings = model(global_audio, torch.tensor([256, 220]), local_audio, torch.full((2, 2), 128))
    assert embeddings.shape == (2, 3, 6)
    losses = LeJEPALoss(knots=5, num_proj=8)(embeddings)
    losses["loss"].backward()
    assert model.encoder.backbone.feature_extractor.conv_blocks[0].conv.weight.grad is not None
    assert model.encoder.backbone.context_encoder.layers[0].attn.qkv.weight.grad is not None
    assert model.projector.net[-1].weight.grad is not None


def test_two_optimizer_steps():
    model = build_audio_lejepa(tiny_cfg())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = LeJEPALoss(knots=5, num_proj=8)
    for _ in range(2):
        embeddings = model(
            torch.randn(2, 256), torch.full((2,), 256),
            torch.randn(2, 2, 128), torch.full((2, 2), 128),
        )
        criterion(embeddings)["loss"].backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def test_collate_keeps_variable_lengths_and_diagnostics():
    samples = []
    for length in (100, 80):
        samples.append({
            "global_audio": torch.ones(length), "global_length": length,
            "local_audio": torch.ones(2, length // 2),
            "local_lengths": torch.tensor([length // 2] * 2),
            "local_intervals": torch.tensor([[0, length // 2]] * 2),
            "anchor_interval": torch.tensor([0, length]), "was_repeated": False,
            "was_padded": length == 80, "augmentation_occurrence": torch.zeros(2, 4),
            "path": str(length),
        })
    batch = collate_audio_lejepa(samples)
    assert batch["global_audio"].shape == (2, 100)
    assert batch["local_audio"].shape == (2, 2, 50)
    assert torch.equal(batch["global_lengths"], torch.tensor([100, 80]))


def test_loader_rejects_audiojepa_training_checkpoint():
    encoder = AudioLeEncoder(tiny_cfg())
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "audiojepa.pt"
        torch.save({"encoder": {}, "target_encoder": {}, "predictor": {}}, path)
        with pytest.raises(ValueError, match="raw emotion2vec"):
            load_emotion2vec_encoder(encoder, path)


def test_checkpoint_resume_and_config_ablation_merge():
    model = build_audio_lejepa(tiny_cfg())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = ModelEMA(model, update_every=1, device="cpu")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "latest.pt"
        save_checkpoint(path, model, optimizer, None, ema, 2, 1, {"test": True})
        restored = build_audio_lejepa(tiny_cfg())
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        restored_ema = ModelEMA(restored, update_every=1, device="cpu")
        checkpoint = load_checkpoint(path, restored, restored_optimizer, None, restored_ema)
    assert checkpoint["step"] == 2
    for left, right in zip(model.parameters(), restored.parameters()):
        assert torch.equal(left, right)

    cfg = load_config("CONFIGS/tasks/pretrain/audio/lejepa/a1-4s-local2s-crop-only.yaml")
    assert cfg["data"]["view_mode"] == "crop_only"
    assert cfg["data"]["sample_rate"] == 16000
