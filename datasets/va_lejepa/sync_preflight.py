"""Check every downstream VA pair once before the training loaders start."""
import json
import fcntl
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from datasets.audio_jepa.pretrain_dataset import load_audio
from .av_decoder import probe_media, validate_sync

LOGGER = logging.getLogger(__name__)


def sync_preflight_path(data_cfg, folder):
    specified = str(data_cfg.get('sync_preflight_file') or '').strip()
    if specified:
        return Path(specified).expanduser().resolve()
    return (Path(folder).expanduser() / 'va_sync_preflight.json').resolve()


def _signature(row):
    return [[str(path), os.stat(path).st_size, os.stat(path).st_mtime_ns]
            for path in (row['video_path'], row['audio_path'])]


def _check_pair(row, sample_rate, sync_cfg, minimum_seconds):
    from decord import VideoReader, cpu

    signature = _signature(row)
    reader = VideoReader(row['video_path'], ctx=cpu(0), num_threads=1)
    fps = float(reader.get_avg_fps())
    if fps <= 0 or len(reader) <= 0:
        raise ValueError('video has no decodable frames or valid frame rate')
    video = probe_media(row['video_path'], 'video')
    video['fps'] = fps
    audio = probe_media(row['audio_path'], 'audio')
    diagnostics = validate_sync(video, audio, row, sync_cfg)
    waveform = load_audio(row['audio_path'], sample_rate)
    decoded_duration = waveform.numel() / sample_rate
    if abs(decoded_duration - audio['duration']) > diagnostics['duration_tolerance_seconds']:
        raise ValueError('decoded audio length differs from probed duration')
    shift = diagnostics['audio_shift_seconds']
    video_origin = max(0., shift)
    common_duration = min(video['duration'], len(reader) / fps, shift + decoded_duration) - video_origin
    if common_duration <= 0 or common_duration < minimum_seconds:
        raise ValueError(f'paired media common duration {common_duration:.3f}s is too short')
    return {'signature': signature, 'video': video, 'audio': audio, 'diagnostics': diagnostics}


def _settings(sample_rate, sync_cfg, minimum_seconds):
    return dict(sample_rate=sample_rate, sync=sync_cfg, minimum_seconds=minimum_seconds)


def _read_cache(path):
    try:
        cache = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    pairs = cache.get('pairs', {})
    if cache.get('version') == 1:
        # Legacy files stored one set of settings for the entire cache.
        settings = _settings(cache.get('sample_rate'), cache.get('sync'), cache.get('minimum_seconds'))
        for entry in pairs.values():
            entry['settings'] = settings
    elif cache.get('version') != 2:
        return {}
    return pairs


def _write_cache(path, pairs):
    payload = {'version': 2, 'pairs': pairs}
    temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    try:
        with temporary.open('w') as handle:
            json.dump(payload, handle)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _matches(entry, row, signature, settings):
    return (entry is not None and entry.get('signature') == signature
            and entry.get('settings') == settings
            and entry['diagnostics'].get('av_offset_seconds', 0.) == float(row.get('av_offset_seconds') or 0))


def _extend_cache(cache_path, rows, signatures, settings, workers):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Independent tasks may add records concurrently to the same mental file.
    # Hold the lock through probing so a pair is checked only once.
    with cache_path.with_name(cache_path.name + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cached = _read_cache(cache_path)
        checked, stale = {}, []
        for pair_id, row in rows.items():
            entry = cached.get(pair_id)
            if _matches(entry, row, signatures[pair_id], settings):
                checked[pair_id] = entry
            else:
                stale.append(pair_id)
        failures = []
        if stale:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_check_pair, rows[pair_id], settings['sample_rate'],
                    settings['sync'], settings['minimum_seconds']): pair_id for pair_id in stale}
                for future in tqdm(as_completed(futures), total=len(futures), desc='VA sync probe'):
                    pair_id = futures[future]
                    try:
                        entry = future.result()
                        entry['settings'] = settings
                        checked[pair_id] = entry
                    except Exception as error:
                        failures.append(f'{pair_id}: {type(error).__name__}: {error}')
            # Preserve every other task's records and keep successful checks
            # even if some pairs fail, so a restart need not repeat them.
            cached.update(checked)
            _write_cache(cache_path, cached)
        if failures:
            raise RuntimeError(f'VA sync preflight failed for {len(failures)} pairs; first errors:\n' + '\n'.join(failures[:20]))
        LOGGER.info('VA sync cache=%s reused=%d newly_checked=%d total_cached=%d',
                    cache_path, len(rows) - len(stale), len(stale), len(cached))
        return checked


def prepare_sync_metadata(datasets, data_cfg, cache_path, rank=0, workers=8):
    """Attach checked metadata to all train/validation rows; rank zero probes media."""
    rows = {row['pair_id']: row for dataset in datasets for row in dataset.rows}
    if sum(len(dataset.rows) for dataset in datasets) != len(rows):
        raise ValueError('duplicate VA pair_id across train and validation splits')
    sample_rate = int(data_cfg.get('sample_rate', 16000))
    sync_cfg = data_cfg.get('sync', {})
    minimum_seconds = float(data_cfg.get('minimum_seconds', .1))
    cache_path = Path(cache_path).expanduser().resolve()
    settings = _settings(sample_rate, sync_cfg, minimum_seconds)

    # Other DDP ranks wait on the shared atomic cache file. They do not enter
    # an NCCL collective while rank zero runs a potentially long media scan.
    while True:
        cache_mtime = cache_path.stat().st_mtime_ns if cache_path.exists() else None
        cached = _read_cache(cache_path)
        checked = {}
        signatures = {}
        stale = []
        for pair_id, row in tqdm(rows.items(), total=len(rows), desc='VA sync preflight', disable=rank != 0):
            try:
                signature = _signature(row)
            except OSError as error:
                LOGGER.error('VA sync preflight pair_id=%s: %s', pair_id, error)
                raise RuntimeError(f'VA sync preflight pair_id={pair_id}: {error}') from error
            signatures[pair_id] = signature
            entry = cached.get(pair_id)
            if _matches(entry, row, signature, settings):
                checked[pair_id] = entry
            else:
                stale.append(pair_id)
        if rank == 0:
            if stale:
                checked = _extend_cache(cache_path, rows, signatures, settings, workers)
            else:
                LOGGER.info('VA sync cache=%s reused=%d newly_checked=0 total_cached=%d',
                            cache_path, len(rows), len(cached))
            break
        if not stale:
            break
        while True:
            time.sleep(2)
            current_mtime = cache_path.stat().st_mtime_ns if cache_path.exists() else None
            if current_mtime != cache_mtime:
                break

    for pair_id, row in rows.items():
        row['_va_sync_metadata'] = checked[pair_id]
    return len(rows)
