# -*- coding: utf-8 -*-

# TransportTools, a library for massive analyses of internal voids in biomolecules and ligand transport through them
# Copyright (C) 2022  Jan Brezovsky <janbre@amu.edu.pl>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

__version__ = '0.9.8'
__author__ = 'Jan Brezovsky'
__mail__ = 'janbre@amu.edu.pl'

import logging
import os
from time import time
from datetime import timedelta
from atexit import register
from argparse import ArgumentParser


logger = logging.getLogger(__name__)

_CONSOLE_HANDLER = None
_LOG_HANDLER = None
_WORKER_LOGFILE = None  # set per spawned-worker process by init_worker_logging() to avoid duplicate handlers
START_TIME = time()
DIVIDER_LINE = "======== ********************************************* ========"


def init_parser() -> ArgumentParser:
    """
    Initiates command line parser
    :return: the parser
    """

    description = "Engine to perform massive analyses of internal voids in biomolecules and ligand transport through " \
                  "them with TransportTool lib. (version {})".format(__version__)
    in_parser = ArgumentParser(description=description)
    group = in_parser.add_mutually_exclusive_group()
    group.add_argument("-c", "--config", dest="config_filename", required=False,
                       help="File with job configuration; runs TransportTool job with the specified configuration.")
    group.add_argument("-w", "--write_template_config", action="store_true", dest="write_config_file", required=False,
                       help="Writes a template job configuration to file 'tmp_config.ini' and exits.")
    in_parser.add_argument("-a", "--advanced", action="store_true", dest="advanced", required=False,
                           help="Enables extension of configuration file ('tmp_config.ini') by advanced section. \
                           This parameter should be used together with '-w' flag.")

    group.add_argument("-v", "--version", action="store_true", dest="print_version", required=False,
                       help="Prints versions and exits.")
    group.add_argument("-l", "--license", action="store_true", dest="print_license", required=False,
                       help="Prints short license info and exits.")
    in_parser.add_argument("--overwrite", action="store_true", dest="overwrite", required=False,
                           help="Enables cleaning of non-empty folder with outputs and overwriting of checkpoints files")

    return in_parser


def progressbar(iteration: int, total: int, log_level: str = "info"):
    """
    Generates progress bar for processes
    :param iteration: current iteration of the process
    :param total: total number of iterations
    :param log_level: logfile logging level restored once the bar is drawn (the logfile handler is
                      muted to WARNING while the bar updates to keep it out of the log file)
    """

    # nothing to draw for an empty workload; bail out before touching the handlers so we neither divide by
    # zero below nor leave the console/log handlers in their muted progress-bar state
    if total <= 0:
        return

    if _CONSOLE_HANDLER is not None and _LOG_HANDLER is not None:
        _CONSOLE_HANDLER.setFormatter(logging.Formatter("\r%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                                        datefmt="%H:%M:%S"))
        _CONSOLE_HANDLER.terminator = ""
        _LOG_HANDLER.setLevel(logging.WARNING)

        length = 60
        fill = '█'
        percent = ("{:0.1f}".format(100 * iteration / total))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)

        if iteration == total:
            _CONSOLE_HANDLER.terminator = "\n"
            logger.info("|{}| {}%".format(bar, percent))
            _CONSOLE_HANDLER.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                                            datefmt="%H:%M:%S"))
        else:
            logger.info("|{}| {}%".format(bar, percent))

        set_logging_level(log_level, _LOG_HANDLER)


def greetings():
    msg = "{:^43}".format("TransportTools execution started")
    logger.info(DIVIDER_LINE)
    logger.info("========  {}  ========".format(msg))
    logger.info(DIVIDER_LINE)


def bye_bye(process_start):
    msg = "{:^43}".format("Overall elapsed time: " + str(timedelta(seconds=(time() - process_start))).split('.')[0])
    logger.info(DIVIDER_LINE)
    logger.info("========  {}  ========".format(msg))
    logger.info(DIVIDER_LINE + "\n\n")


def license_printer():
    """
    Prints info about the license
    """

    print("""
# TransportTools, a library for massive analyses of internal voids in biomolecules and ligand transport through them
# Copyright (C) 2022  Jan Brezovsky <janbre@amu.edu.pl>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.""")


class _VerboseFilter(logging.Filter):
    """Drops the high-volume per-step DEBUG chatter from the logfile unless verbose logging was requested."""

    def filter(self, record):
        if record.levelno == logging.DEBUG:
            msg = record.getMessage()
            if msg.startswith("Optimizing assignment"):
                return False
            if msg.startswith("Distance matrix"):
                return False
            if msg.startswith("Using point") or msg.startswith("No points") or \
                    msg.startswith("Using starting point"):
                return False
            if "alignment_length" in msg or "Using CA atoms" in msg or "General rotation matrix" in msg:
                return False
            if "max_dist = " in msg and "layer_thickness =" in msg:
                return False
            if msg.startswith("Transport event") or msg.startswith("Optimized distance"):
                return False
        return True


class _DefaultFilter(logging.Filter):
    """Always-on noise filter for the logfile (e.g. matplotlib font probing)."""

    def filter(self, record):
        if "findfont: " in record.getMessage():
            return False
        return True


def _build_logfile_handler(verbose_logging: bool, logfile: str) -> logging.FileHandler:
    """
    Build the logfile handler shared by the driver and the spawned workers: identical formatter, DEBUG
    ceiling and noise filters, opened in append mode so several processes can write to the same file.
    :param verbose_logging: if more details should be provided on debug level
    :param logfile: file to log into
    """

    fh = logging.FileHandler(logfile)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    fh.setLevel(logging.DEBUG)
    if not verbose_logging:
        fh.addFilter(_VerboseFilter())
    fh.addFilter(_DefaultFilter())
    return fh


def init_logging(verbose_logging: bool = False, logfile: str = "transport_tools.log"):
    """
    Initiates and sets logging, also defines logging filtering
    :param verbose_logging: if more details should be provided on debug level
    :param logfile: file to log into
    """

    global _CONSOLE_HANDLER
    global _LOG_HANDLER
    fh = _build_logfile_handler(verbose_logging, logfile)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"))
    ch.setLevel(logging.INFO)

    logging.getLogger().addHandler(ch)
    logging.getLogger().setLevel(logging.NOTSET)
    logging.getLogger().addHandler(fh)
    _CONSOLE_HANDLER = ch
    _LOG_HANDLER = fh


def init_worker_logging(log_level: str, verbose_logging: bool, logfile: str):
    """
    Route logging from a spawned worker process to the driver's logfile.

    Workers started under the 'spawn' start method inherit none of the parent's logging handlers, so any
    logger.debug()/info() they emit (e.g. the per-event ambiguous-assignment resolution decisions) would
    otherwise be discarded. This attaches a logfile-only handler (no console handler, so workers do not each
    duplicate stdout) identical to the driver's and raises the root level so records reach it. The logfile is
    opened in append mode; each record is written by a single write, so records stay intact across processes
    and only their relative ordering is non-deterministic. Idempotent per worker process.
    :param log_level: logging level to be used for the logfile
    :param verbose_logging: if more details should be provided on debug level
    :param logfile: file to log into (the same logfile as the driver)
    """

    global _WORKER_LOGFILE
    if _WORKER_LOGFILE == logfile:
        return
    fh = _build_logfile_handler(verbose_logging, logfile)
    set_logging_level(log_level, fh)
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.NOTSET)
    _WORKER_LOGFILE = logfile


def set_logging_level(level: str, handler):
    """
    Sets currently used level of logging
    :param level: logging level to be used
    :param handler: the logging handler whose level is set (the console or the logfile handler)
    """

    level = level.upper()

    mapping = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARN": logging.WARNING,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }

    if handler is None:
        raise RuntimeError("Logging has not been initialized. Call init_logging() first.")
    handler.setLevel(mapping.get(level, logging.NOTSET))


def process_count(num_processes: int) -> str:
    if num_processes == 1:
        return "parallel process"
    else:
        return "parallel processes"


def initiate_tools(std_level: str = "info", log_level: str = "info", verbose_logging: bool = False, logfile: str = "transport_tools.log"):
    """
    Starts logging, enables initial and terminal messages
    :param std_level: logging level to be used for std
    :param log_level: logging level to be used for logfile
    :param verbose_logging: if more details should be provided on debug level
    :param logfile: file to log into
    """

    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    init_logging(verbose_logging, logfile)
    set_logging_level(std_level, _CONSOLE_HANDLER)
    set_logging_level(log_level, _LOG_HANDLER)
    greetings()
    register(bye_bye, START_TIME)


class SuppressMsg:
    def __init__(self):
        """
        A context manager to suppress all messages even those originating from external programs
        but not suppressing exceptions.
        """

        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


class TimeProcess:
    def __init__(self, prefix_msg: str = ""):
        """
        Monitors and reports duration of the process(es)
        :param prefix_msg: text preceding the report on the process duration
        """

        self.prefix_msg = prefix_msg

    def __enter__(self):
        self.initial_time = time()

    def __exit__(self, *_):
        elapsed_time = timedelta(seconds=(time() - self.initial_time))
        logger.info("{} took: {}.".format(self.prefix_msg, str(elapsed_time).split(".")[0]))
