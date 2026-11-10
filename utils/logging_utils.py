"""Rank-filtered logging for Slurm jobs."""

import os
import logging

_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def slurm_filter(record):
    """Keep log records only on Slurm process zero."""
    return int(os.environ["SLURM_PROCID"]) == 0


def config_logger(log_level=logging.INFO):
    """Configure the root logger and keep output on Slurm process zero."""
    logging.basicConfig(format=_format, level=log_level)
    root_logger = logging.getLogger()
    root_logger.addFilter(slurm_filter)


def log_to_file(
    logger_name=None, log_level=logging.INFO, log_filename="tensorflow.log", mode="w"
):
    """Attach a file handler to the named logger, or to the root logger."""

    if not os.path.exists(os.path.dirname(log_filename)):
        os.makedirs(os.path.dirname(log_filename))

    if logger_name is not None:
        log = logging.getLogger(logger_name)
    else:
        log = logging.getLogger()

    fh = logging.FileHandler(log_filename, mode=mode)
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter(_format))
    log.addHandler(fh)


def log_versions():
    """Log the PyTorch build and selected library versions."""
    import torch
    import subprocess

    logging.info("--------------- Versions ---------------")
    logging.info(
        "git branch: " + str(subprocess.check_output(["git", "branch"]).strip())
    )
    logging.info(
        "git hash: "
        + str(subprocess.check_output(["git", "rev-parse", "HEAD"]).strip())
    )
    logging.info("Torch: " + str(torch.__version__))
    logging.info("----------------------------------------")


class disable_logging(object):
    """
    A context manager to disable logging temporarily.
    """

    def __init__(self, level=logging.ERROR):  # pragma: no cover
        """
        Initialize the context manager.
        """
        logging.disable(level=level)

    def __enter__(self):  # pragma: no cover
        """
        Enter the context manager.
        """
        return self

    def __exit__(self, type, value, traceback):  # pragma: no cover
        """
        Exit the context manager and enable logging.
        """
        logging.disable(level=logging.NOTSET)
