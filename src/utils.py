import os
import shutil
from pathlib import Path
from contextlib import contextmanager


@contextmanager
def temporary_file_change(file_path: str | os.PathLike | Path):
    current_file = Path(file_path)
    current_dir = file_path.parent
    taken_files: list[str] = [file for file in os.listdir(current_dir) if 'real_file' in file]
    if len(taken_files):
        last_taken_digit = int(sorted(taken_files)[-1].replace('real_file', ''))
    else:
        last_taken_digit = 0
    real_file = current_file.parent.joinpath(f'real_file{last_taken_digit + 1}')
    shutil.copy(current_file, real_file)
    yield
    shutil.move(real_file, current_file)


@contextmanager
def pushd(directory):
    cur_dir = os.getcwd()
    os.chdir(directory)
    yield
    os.chdir(cur_dir)
