"""Shared baseline and FaVoR checks for usable audio-video pairs."""
import json
import csv
import hashlib
import logging
import subprocess
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch.distributed as dist

LOGGER = logging.getLogger(__name__)

_SOUNDFILE_SUFFIXES = {'.wav', '.flac', '.ogg', '.aif', '.aiff', '.au'}
# Matches the default data.minimum_seconds used by the FaVoR sync preflight.
MINIMUM_SECONDS = 0.1


def _media_readable(filename, video):
    target = Path(filename)
    try:
        if target.stat().st_size < 1024:
            return False
    except OSError:
        return False
    if video:
        from decord import VideoReader, cpu
        try:
            return len(VideoReader(str(target), ctx=cpu(0), num_threads=1)) > 0
        except Exception:
            return False
    if target.suffix.lower() not in _SOUNDFILE_SUFFIXES:
        return True
    import soundfile
    try:
        return soundfile.info(str(target)).frames > 0
    except Exception:
        return False


def _ffprobe_duration(filename):
    result = subprocess.run(('ffprobe', '-v', 'error', '-select_streams', 'a:0',
                             '-show_entries', 'stream=duration', '-of', 'json', str(filename)),
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30)
    result.check_returncode()
    streams = json.loads(result.stdout.decode()).get('streams') or []
    return float(streams[0]['duration'])


def _sync_ok(audio_path, video_path, minimum_seconds=MINIMUM_SECONDS):
    from decord import VideoReader, cpu
    try:
        reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        fps = float(reader.get_avg_fps())
        video_duration = len(reader) / fps
        if Path(audio_path).suffix.lower() in _SOUNDFILE_SUFFIXES:
            import soundfile
            info = soundfile.info(str(audio_path))
            audio_duration = info.frames / info.samplerate
        else:
            audio_duration = _ffprobe_duration(audio_path)
        if min(video_duration, audio_duration) < minimum_seconds:
            return False
        return abs(video_duration - audio_duration) <= max(0.5, 2 / fps)
    except Exception:
        return False


def paired_media_readable(row, minimum_seconds=MINIMUM_SECONDS):
    return (_media_readable(row['audio_path'], False)
            and _sync_ok(row['audio_path'], row['video_path'], minimum_seconds))


def filter_paired_datasets(datasets, report_path, workers=8, minimum_seconds=MINIMUM_SECONDS):
    """Filter train/evaluation rows once and share the decision with all ranks."""
    rows = [(split, row) for split, dataset in datasets for row in dataset.rows]
    ids = [row['pair_id'] for _, row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate VA pair_id across training and evaluation splits')
    digest = hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    result = [None]
    if rank == 0:
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                valid = list(pool.map(partial(paired_media_readable, minimum_seconds=minimum_seconds), (row for _, row in rows)))
            rejected = [(split, row) for (split, row), ok in zip(rows, valid) if not ok]
            report_path = Path(report_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=['split', 'pair_id', 'audio_path', 'video_path'])
                writer.writeheader()
                writer.writerows(dict(split=split, **{k: row[k] for k in ('pair_id', 'audio_path', 'video_path')})
                                 for split, row in rejected)
            result[0] = {'digest': digest, 'excluded': [row['pair_id'] for _, row in rejected]}
        except Exception as error:
            result[0] = {'error': f'{type(error).__name__}: {error}'}
    if distributed:
        dist.broadcast_object_list(result, src=0)
    if 'error' in result[0]:
        raise RuntimeError(f'VA pair filtering failed: {result[0]["error"]}')
    if result[0]['digest'] != digest:
        raise RuntimeError('VA pair filtering: dataset rows differ across ranks')
    excluded = set(result[0]['excluded'])
    for split, dataset in datasets:
        kept = [i for i, row in enumerate(dataset.rows) if row['pair_id'] not in excluded]
        removed = len(dataset.rows) - len(kept)
        dataset.rows = [dataset.rows[i] for i in kept]
        dataset.labels = [dataset.labels[i] for i in kept]
        if rank == 0:
            LOGGER.info('VA pair filter split=%s kept=%d skipped=%d report=%s',
                        split, len(kept), removed, report_path)
        if not kept:
            raise ValueError(f'no usable VA pairs in {split} after filtering')
