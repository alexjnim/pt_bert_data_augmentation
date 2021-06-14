import pandas as pd
from config import config
from pathlib import Path

print(config.aug_file_path)


# if Path(config.aug_file_path).is_file():
#     print("loading data...")


fpath = Path("").absolute()
# fpath = Path("").resolve().parent

print(fpath)

print(fpath / "path_test.py")
