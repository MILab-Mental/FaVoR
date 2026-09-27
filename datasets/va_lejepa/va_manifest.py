"""Canonical paired manifests with explicit identity verification."""
import csv
import hashlib
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


def iter_manifest(path, *, root=None, source_pattern=None, progress=False):
    path = Path(path)
    with path.open(encoding='utf-8-sig', newline='') as handle, progress_lines(handle, path, 'Read manifest', progress) as lines:
        first = handle.readline()
        handle.seek(0)
        reader = csv.DictReader(lines, delimiter='\t' if '\t' in first else ',')
        canonical = bool(reader.fieldnames and REQUIRED.issubset(reader.fieldnames))
        if not reader.fieldnames or not {'audio_path', 'video_path'}.issubset(reader.fieldnames):
            raise ValueError(f'{path}: expected canonical columns {sorted(REQUIRED)}; run manifest_tools convert')
        seen = set()
        count = 0
        parents = {}
        for line, row in enumerate(reader, 2):
            try:
                if not canonical:
                    if not row.get('audio_path', '').strip() or not row.get('video_path', '').strip():
                        continue
                    row['source_id'] = ''
                    row['pair_id'] = ''
                for field in REQUIRED:
                    row[field] = str(row[field] or '').strip()
                for field in ('video_path', 'audio_path'):
                    value = Path(row[field]).expanduser()
                    if not value.is_absolute():
                        value = Path(root or path.parent) / value
                    # Resolving the shared parent once avoids repeated walks
                    # through the same directories for millions of files.
                    if value.name == '..':
                        resolved = value.resolve()
                    else:
                        if value.parent not in parents:
                            if len(parents) >= 8192:
                                parents.clear()
                            parents[value.parent] = value.parent.resolve()
                        resolved = parents[value.parent] / value.name
                        if resolved.is_symlink():
                            resolved = resolved.resolve()
                    row[field] = str(resolved)
                if not canonical:
                    row['source_id'] = source_key(row['video_path'], source_pattern)
                    row['pair_id'] = hashlib.sha256(f'{row["video_path"]}\0{row["audio_path"]}'.encode()).hexdigest()[:24]
                validate_identity(row, source_pattern)
                if row['pair_id'] in seen:
                    raise ValueError(f'duplicate pair_id {row["pair_id"]}')
                seen.add(row['pair_id'])
                count += 1
                yield row
            except Exception as error:
                raise ValueError(f'{path}:{line}: {error}') from error
    if not count:
        raise ValueError(f'{path}: empty manifest')


def read_manifest(path, *, root=None, source_pattern=None, progress=False):
    return list(iter_manifest(path, root=root, source_pattern=source_pattern, progress=progress))
