"""Canonical paired manifests with explicit identity verification."""
import csv
import re
from pathlib import Path
from .progress import progress_lines

REQUIRED = {'pair_id', 'source_id', 'video_path', 'audio_path'}


def source_key(path, pattern=None):
    stem = Path(path).stem
    if pattern:
        match = re.fullmatch(pattern, stem)
        if not match:
            raise ValueError(f'cannot extract source identity from {path}')
        return match.group('source_id') if 'source_id' in match.groupdict() else match.group(1)
    # RAVDESS's first field is modality (01=AV, 02=video, 03=audio),
    # not recording identity. Scope this rule to the dataset directory.
    # Official naming convention: https://zenodo.org/records/1188976
    if any(part.casefold() == 'ravdess' for part in Path(path).parts):
        match = re.fullmatch(r'0[123]-((?:\d{2}-){5}\d{2})', stem)
        if match:
            return 'ravdess:' + match.group(1)
    return stem


def validate_identity(row, pattern=None):
    video = source_key(row['video_path'], pattern)
    audio = source_key(row['audio_path'], pattern)
    if not row['source_id'] or video != audio or video != row['source_id']:
        raise ValueError(f'pair {row["pair_id"]}: source identity mismatch ({video}, {audio}, {row["source_id"]})')
    if not row['pair_id']:
        raise ValueError('empty pair_id')


def read_manifest(path, *, root=None, source_pattern=None, progress=False):
    path = Path(path)
    with path.open(encoding='utf-8-sig', newline='') as handle, progress_lines(handle, path, 'Read manifest', progress) as lines:
        first = handle.readline()
        handle.seek(0)
        reader = csv.DictReader(lines, delimiter='\t' if '\t' in first else ',')
        if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
            raise ValueError(f'{path}: expected canonical columns {sorted(REQUIRED)}; run manifest_tools convert')
        rows, seen = [], set()
        for line, row in enumerate(reader, 2):
            try:
                for field in REQUIRED:
                    row[field] = str(row[field] or '').strip()
                for field in ('video_path', 'audio_path'):
                    value = Path(row[field]).expanduser()
                    if not value.is_absolute():
                        value = Path(root or path.parent) / value
                    row[field] = str(value.resolve())
                validate_identity(row, source_pattern)
                if row['pair_id'] in seen:
                    raise ValueError(f'duplicate pair_id {row["pair_id"]}')
                seen.add(row['pair_id'])
                rows.append(row)
            except Exception as error:
                raise ValueError(f'{path}:{line}: {error}') from error
    if not rows:
        raise ValueError(f'{path}: empty manifest')
    return rows
