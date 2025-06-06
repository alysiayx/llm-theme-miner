import logging
import sys
from pathlib import Path
from typing import Union
from datetime import datetime

from .paths import LOGS_DIR

def get_logger(name: str = "theme_extraction", log_dir: Union[str, Path, None] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger 

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_path = Path(log_dir) if log_dir else LOGS_DIR
    log_path.mkdir(parents=True, exist_ok=True)
    date = datetime.today().strftime('%Y%m%d')
    file_handler = logging.FileHandler(log_path / f"{name}-{date}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger