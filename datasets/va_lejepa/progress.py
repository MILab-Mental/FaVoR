"""File progress without a separate full-file counting pass."""
from contextlib import contextmanager
from pathlib import Path
from tqdm import tqdm


@contextmanager
def progress_lines(handle, path, description, enabled):
    def tracked(bar):
        for line in handle:
            bar.update(len(line.encode('utf-8')))
            yield line

    with tqdm(total=Path(path).stat().st_size, desc=description, unit='B',
              unit_scale=True, dynamic_ncols=True, disable=not enabled) as bar:
        yield tracked(bar) if enabled else handle
