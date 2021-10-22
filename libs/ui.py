# -*- coding: utf-8 -*-

# TransportTools, a library for massive analyses of internal voids in biomolecules and ligand transport through them
# Copyright (C) 2021  Jan Brezovsky, Aravind Selvaram Thirunavukarasu, Carlos Eduardo Sequeiros-Borja, Bartlomiej
# Surpeta, Nishita Mandal, Cedrix Jurgal Dongmo Foumthuim, Dheeraj Kumar Sarkar, Nikhil Agrawal  <janbre@amu.edu.pl>
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

__version__ = '0.9.0'
__author__ = 'Jan Brezovsky, Aravind Selvaram Thirunavukarasu, Carlos Eduardo Sequeiros-Borja, Bartlomiej Surpeta, ' \
             'Nishita Mandal, Cedrix Jurgal Dongmo Foumthuim, Dheeraj Kumar Sarkar, Nikhil Agrawal'
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


def progressbar(iteration: int, total: int):
    """
    Generates progress bar for processes
    :param iteration: current iteration of the process
    :param total: total number of iterations
    """

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

        _LOG_HANDLER.setLevel(logging.DEBUG)


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
# Copyright (C) 2021  Jan Brezovsky, Aravind Selvaram Thirunavukarasu, Carlos Eduardo Sequeiros-Borja, Bartlomiej
# Surpeta, Nishita Mandal, Cedrix Jurgal Dongmo Foumthuim, Dheeraj Kumar Sarkar, Nikhil Agrawal  <janbre@amu.edu.pl>
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


def init_logging(verbose_logging: bool = False, logfile: str = "transport_tools.log"):
    """
    Initiates and sets logging, also defines logging filtering
    :param verbose_logging: if more details should be provided on debug level
    :param logfile: file to log into
    """

    global _CONSOLE_HANDLER
    global _LOG_HANDLER
    fh = logging.FileHandler(logfile)
    ch = logging.StreamHandler()

    ch_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    fh_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                     datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fh_formatter)
    ch.setFormatter(ch_formatter)

    fh.setLevel(logging.DEBUG)
    ch.setLevel(logging.INFO)

    if not verbose_logging:
        class VerboseFilter(logging.Filter):
            def filter(self, record):
                if record.levelno == logging.DEBUG:
                    if record.getMessage().startswith("Optimizing assignment"):
                        return False
                    if record.getMessage().startswith("Distance matrix"):
                        return False
                    if record.getMessage().startswith("Using point") or record.getMessage().startswith("No points") or \
                            record.getMessage().startswith("Using starting point"):
                        return False
                    if "alignment_length" in record.getMessage() or "Using CA atoms" in record.getMessage() or \
                            "General rotation matrix" in record.getMessage():
                        return False
                    if "max_dist = " in record.getMessage() and "layer_thickness =" in record.getMessage():
                        return False
                    if record.getMessage().startswith("Transport event") or \
                            record.getMessage().startswith("Optimized distance"):
                        return False
                return True

        fh.addFilter(VerboseFilter())

    class DefaultFilter(logging.Filter):
        def filter(self, record):
            if "findfont: " in record.getMessage():
                return False
            return True

    fh.addFilter(DefaultFilter())

    logging.getLogger().addHandler(ch)
    logging.getLogger().setLevel(logging.NOTSET)
    logging.getLogger().addHandler(fh)
    _CONSOLE_HANDLER = ch
    _LOG_HANDLER = fh


def set_logging_level(level: str):
    """
    Sets currently used level of logging
    :param level: logging level to be used
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

    _CONSOLE_HANDLER.setLevel(mapping.get(level, logging.NOTSET))


def process_count(num_processes: int) -> str:
    if num_processes == 1:
        return "parallel process"
    else:
        return "parallel processes"


def initiate_tools(level: str = "info", verbose_logging: bool = False, logfile: str = "transport_tools.log"):
    """
    Starts logging, enables initial and terminal messages
    :param level: logging level to be used
    :param verbose_logging: if more details should be provided on debug level
    :param logfile: file to log into
    """

    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    init_logging(verbose_logging, logfile)
    set_logging_level(level)
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
