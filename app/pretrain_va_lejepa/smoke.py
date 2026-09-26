"""Two-step hardware calibration with real paired data and checkpoint resume.

Run with torchrun --nproc_per_node=2 -m app.pretrain_va_lejepa.smoke
Use --manifest pointing to a small canonical CSV of strictly valid >=6s pairs.
"""
import argparse
import json
import time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from app.main import load_config
from app.pretrain_va_lejepa.train import _optimizer
from datasets.va_lejepa import VALeJEPADataset, collate_va_lejepa, to_device
from models.va_lejepa import build_va_lejepa, VABranchLoss, save_checkpoint, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='CONFIGS/tasks/pretrain/audio-video/lejepa/i0.yaml')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--video-checkpoint', required=True)
    parser.add_argument('--audio-checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--locals', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--modes', nargs='+', default=['feature'])
    args = parser.parse_args()
    distributed = 'RANK' in __import__('os').environ
    if distributed:
        dist.init_process_group('nccl')
    rank = dist.get_rank() if distributed else 0
    device = torch.device('cuda', int(__import__('os').environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    config = load_config(args.config)
    config['model']['video'].update(init_source='vjepa', vjepa_checkpoint=args.video_checkpoint)
    config['model']['audio'].update(init_source='emotion2vec', emotion2vec_checkpoint=args.audio_checkpoint)
    config['model']['fusion']['enabled'] = args.modes
    config['data'].update(manifests=[args.manifest], local_views=args.locals)
    model = build_va_lejepa(config['model'], config['data'], initialize=rank == 0).to(device)
    if distributed:
        model = DDP(model, device_ids=[device.index])
    raw = model.module if distributed else model
    optimizer = _optimizer(raw, config['optimization'])
    criterion = VABranchLoss(raw, config['loss']).to(device)
    dataset = VALeJEPADataset(config['data'], training=False)
    samples = [dataset[(rank * args.batch_size + i) % len(dataset)] for i in range(args.batch_size)]
    batch = to_device(collate_va_lejepa(samples), device)
    measurements = []
    for step in range(2):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(batch)
            losses = criterion(output)
        torch.cuda.synchronize()
        forward_seconds = time.perf_counter() - started
        forward_peak = torch.cuda.max_memory_allocated()
        started_backward = time.perf_counter()
        losses['loss'].backward()
        torch.cuda.synchronize()
        backward_seconds = time.perf_counter() - started_backward
        backward_peak = torch.cuda.max_memory_allocated()
        assert all(any(p.grad is not None and p.grad.isfinite().all() for p in module.parameters())
                   for module in (raw.video_encoder, raw.audio_encoder, raw.fusion_adapters))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        optimizer_peak = torch.cuda.max_memory_allocated()
        measurements.append(dict(step=step+1, loss=float(losses['loss']), forward_seconds=forward_seconds,
            backward_seconds=backward_seconds, step_seconds=time.perf_counter()-started,
            forward_peak_bytes=forward_peak, backward_peak_bytes=backward_peak, optimizer_peak_bytes=optimizer_peak))
    # Log alignment of one real view, including exact audio bin counts and media provenance.
    raw.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        _, aligned = raw.encode_view(batch['global'], return_alignment=True)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    report = dict(rank=rank, hardware=torch.cuda.get_device_name(), batch_size=args.batch_size, locals=args.locals,
        modes=args.modes, measurements=measurements, pair_ids=batch['pair_id'], video_paths=batch['video_path'], audio_paths=batch['audio_path'],
        diagnostics=batch['sync_diagnostics'], global_intervals=torch.stack((batch['global']['start_time'],batch['global']['end_time']),1).tolist(),
        local_intervals=torch.stack((batch['local']['start_time'],batch['local']['end_time']),2).tolist() if args.locals else [],
        video_timestamps=batch['global']['video_frame_times'].tolist(),audio_sample_ranges=batch['global']['audio_sample_range'].tolist(),
        Tv=aligned.video.temporal_size,Ta=aligned.audio.temporal_size,audio_bin_counts=aligned.audio_counts.tolist())
    (output_path / f'rank-{rank}.json').write_text(json.dumps(report,indent=2))
    if rank == 0:
        save_checkpoint(output_path/'latest.pt',raw,optimizer,step=2,config=config)
    if distributed:
        dist.barrier()
    checkpoint = load_checkpoint(output_path/'latest.pt',raw,optimizer)
    assert checkpoint['step'] == 2
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    print(json.dumps(dict(rank=rank,passed=True,report=str(output_path/f'rank-{rank}.json'))))


if __name__ == '__main__':
    main()
