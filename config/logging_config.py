import logging
import sys
import pathlib
from config import config

# formatter documentation
# https://docs.python.org/3/library/logging.html#logrecord-attributes

PROJECT_PATH = pathlib.Path("").absolute()
LOG_DIR = "logs"
LOG_FILE_NAME = (
    "saved_logs_pta_"
    + str(config.percent_to_augment)
    + "_rf_"
    + str(config.reduce_factor)
    + "_nsps_"
    + str(config.new_sent_per_sent)
    + "_nwr_"
    + str(config.num_words_replace)
    + "_ts_"
    + str(config.test_size)
    + ".log"
)

FORMATTER = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s —" "%(funcName)s:%(lineno)d — \n%(message)s"
)

# in order of increasing severity, are the following:
# DEBUG
# INFO
# WARNING
# ERROR
# CRITICAL

file_handler = logging.FileHandler(PROJECT_PATH / LOG_DIR / LOG_FILE_NAME, mode="w")
file_handler.setFormatter(FORMATTER)
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(FORMATTER)
console_handler.setLevel(logging.INFO)


def configure_logger(logger):
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    # it's not enough to define a handler with level=DEBUG, the actual logging level must also be DEBUG in order to get it to output anything.
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger
