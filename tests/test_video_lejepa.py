import csv
import tempfile
import importlib.util
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from app.pretrain_video_lejepa.train import _trim_history, load_training_checkpoint, save_checkpoint
from datasets.video_lejepa import VideoLeJEPAMultiCrop, collate_video_lejepa
from models.video_lejepa import LeJEPALoss, ModelEMA, Projector, VideoLeJEPA, load_vjepa_encoder
from models.video_jepa.vision_transformer import VisionTransformer, build_block_causal_mask


def tiny_model(token_drop_rate=0.5, attn_mode="block_causal"):
    encoder = VisionTransformer(
        img_size=16,
        patch_size=8,
        num_frames=2,
        tubelet_size=1,
        embed_dim=24,
        depth=1,
        num_heads=3,
        use_rope=True,
        use_cls_token=True,
        token_drop_rate=token_drop_rate,
        attn_mode=attn_mode,
    )
    return VideoLeJEPA(encoder, Projector(24, hidden_dim=16, output_dim=8))


def test_trim_history_removes_unsaved_epoch_and_reused_accumulation_step(tmp_path):
    history = tmp_path / "history.csv"
    with history.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "iteration", "global_step"])
        writer.writeheader()
        writer.writerows([
            {"epoch": 1, "iteration": 0, "global_step": 1},
            {"epoch": 1, "iteration": 1, "global_step": 2},
            {"epoch": 2, "iteration": 0, "global_step": 2},
            {"epoch": 2, "iteration": 1, "global_step": 3},
        ])

    _trim_history(history, checkpoint_epoch=1, checkpoint_step=2)

    with history.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(int(row["epoch"]), int(row["global_step"])) for row in rows] == [(1, 1), (1, 2)]


def test_multicrop_shapes_and_temporal_contract():
    frames = torch.randint(0, 256, (4, 40, 48, 3), dtype=torch.uint8)
    transform = VideoLeJEPAMultiCrop(global_size=32, local_size=16, local_views=3)
    samples = []
    for name in ("a", "b"):
        sample = transform(frames)
        sample.update(frame_indices=torch.arange(4), path=name)
        samples.append(sample)
    batch = collate_video_lejepa(samples)
    assert batch["global_video"].shape == (2, 3, 4, 32, 32)
    assert batch["local_video"].shape == (2, 3, 3, 4, 16, 16)
    assert torch.equal(batch["frame_indices"][0], batch["frame_indices"][1])


def test_cls_token_drop_and_eval_full_tokens():
    model = tiny_model().encoder
    video = torch.randn(2, 3, 2, 16, 16)
    model.train()
    assert model(video).shape == (2, 5, 24)  # CLS + half of 8 patches
    model.eval()
    assert model(video).shape == (2, 9, 24)
    assert model(video, return_tokens=False).shape == (2, 24)


def test_block_causal_mask_has_no_future_or_cls_leak():
    mask = build_block_causal_mask(2, 1, 2, num_prefix_tokens=1)[0, 0]
    assert mask[0].all()  # CLS reads all tokens
    assert not mask[1:, 0].any()  # patches cannot use CLS as a future-information relay
    assert mask[1, 2] and not mask[1, 3]  # same frame allowed; next frame denied
    assert mask[3, 1]  # later frame can read earlier frame


def test_sigreg_and_total_loss_backward_without_global_detach():
    embeddings = torch.randn(3, 4, 8, requires_grad=True)
    losses = LeJEPALoss(sigreg_weight=0.02, knots=5, num_proj=16)(embeddings)
    losses["loss"].backward()
    assert embeddings.grad is not None
    assert embeddings.grad[:, 0].abs().sum() > 0


def test_reference_levjepa_projector_and_loss_parity():
    spec = importlib.util.spec_from_file_location("levjepa_reference_module", "/home/data/sdc/LeVJEPA/module.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    ours = Projector(8, hidden_dim=16, output_dim=6)
    theirs = reference.Projector(8, hidden_dim=16, output_dim=6)
    theirs.load_state_dict(ours.state_dict())
    ours.eval()
    theirs.eval()
    features = torch.randn(3, 4, 8)
    ours_projection = ours(features)
    reference_projection = theirs(features)
    assert torch.equal(ours_projection, reference_projection)

    torch.manual_seed(77)
    ours_losses = LeJEPALoss(sigreg_weight=0.02, knots=5, num_proj=16)(ours_projection)
    torch.manual_seed(77)
    reference_sigreg = reference.SIGReg(knots=5, num_proj=16)(
        reference_projection.permute(1, 0, 2)
    )
    reference_invariance = (reference_projection[:, :1] - reference_projection).pow(2).mean()
    assert torch.allclose(ours_losses["invariance_loss"], reference_invariance)
    assert torch.allclose(ours_losses["sigreg_loss"], reference_sigreg)
    assert torch.allclose(ours_losses["loss"], reference_invariance + 0.02 * reference_sigreg)


def _ddp_sigreg_worker(rank, init_file, embeddings, result_file):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        torch.manual_seed(123)
        local = embeddings[rank * 2 : (rank + 1) * 2].clone().requires_grad_(True)
        loss = LeJEPALoss(sigreg_weight=1.0, knots=5, num_proj=16)(local)["sigreg_loss"]
        loss.backward()
        torch.save((loss.detach(), local.grad.detach()), f"{result_file}.{rank}")
    finally:
        dist.destroy_process_group()


def test_ddp_sigreg_matches_global_batch():
    embeddings = torch.randn(4, 3, 8)
    with tempfile.TemporaryDirectory() as directory:
        init_file = str(Path(directory) / "init")
        result_file = str(Path(directory) / "result.pt")
        mp.start_processes(
            _ddp_sigreg_worker,
            args=(init_file, embeddings, result_file),
            nprocs=2,
            join=True,
            start_method="spawn",
        )
        rank0_loss, rank0_grad = torch.load(f"{result_file}.0", weights_only=True)
        rank1_loss, rank1_grad = torch.load(f"{result_file}.1", weights_only=True)
    # The collective ECF makes the scalar objective identical on every rank,
    # while both local shards retain a usable backward path.
    assert torch.equal(rank0_loss, rank1_loss)
    assert torch.isfinite(rank0_grad).all() and torch.isfinite(rank1_grad).all()
    torch.manual_seed(123)
    global_loss = LeJEPALoss(sigreg_weight=1.0, knots=5, num_proj=16)(embeddings)["sigreg_loss"]
    assert torch.allclose(rank0_loss, global_loss, atol=1e-5, rtol=1e-5)


def test_vjepa_checkpoint_conversion_and_ratio():
    source = VisionTransformer(
        img_size=16, patch_size=8, num_frames=2, tubelet_size=2,
        embed_dim=24, depth=1, num_heads=3, use_rope=True,
    )
    target = tiny_model(token_drop_rate=0.95).encoder
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "source.pt"
        torch.save({"encoder": {f"module.backbone.{k}": v for k, v in source.state_dict().items()}}, path)
        result = load_vjepa_encoder(target, path, min_load_ratio=0.9)
    assert result["loaded_ratio"] > 0.99
    expected = source.patch_embed.proj.weight.mean(2, keepdim=True)
    assert torch.equal(target.patch_embed.proj.weight, expected)


def test_vjepa_absolute_position_conversion_adds_cls():
    source = VisionTransformer(
        img_size=16, patch_size=8, num_frames=2, tubelet_size=1,
        embed_dim=24, depth=1, num_heads=3, use_rope=False,
    )
    target = VisionTransformer(
        img_size=16, patch_size=8, num_frames=2, tubelet_size=1,
        embed_dim=24, depth=1, num_heads=3, use_rope=False,
        use_cls_token=True, token_drop_rate=0.95, attn_mode="block_causal",
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "source.pt"
        torch.save({"encoder": source.state_dict()}, path)
        result = load_vjepa_encoder(target, path, min_load_ratio=0.9)
    assert "pos_embed" not in result["missing"]
    assert torch.equal(target.pos_embed[:, 0], torch.zeros_like(target.pos_embed[:, 0]))
    assert torch.equal(target.pos_embed[:, 1:], source.pos_embed)


def test_two_optimizer_steps_and_resume():
    torch.manual_seed(0)
    model = tiny_model(token_drop_rate=0.0, attn_mode="full")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = LeJEPALoss(knots=5, num_proj=8)
    ema = ModelEMA(model, update_every=1, device="cpu")
    for step in (1, 2):
        global_video = torch.randn(2, 3, 2, 16, 16)
        local_video = torch.randn(2, 2, 3, 2, 16, 16)
        loss = criterion(model(global_video, local_video))["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        ema.update(model, step)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "latest.pt"
        save_checkpoint(path, model, optimizer, None, ema, 2, 2, {"test": True})
        restored = tiny_model(token_drop_rate=0.0, attn_mode="full")
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        restored_ema = ModelEMA(restored, update_every=1, device="cpu")
        epoch, step, _ = load_training_checkpoint(path, restored, restored_optimizer, None, restored_ema)
    assert (epoch, step) == (2, 2)
    for left, right in zip(model.parameters(), restored.parameters()):
        assert torch.equal(left, right)


def test_legacy_vjepa_defaults_have_no_cls_and_keep_all_patches():
    model = VisionTransformer(
        img_size=16, patch_size=8, num_frames=2, tubelet_size=1,
        embed_dim=24, depth=1, num_heads=3, use_rope=True,
    )
    assert "cls_token" not in model.state_dict()
    assert model(torch.randn(1, 3, 2, 16, 16)).shape == (1, 8, 24)
