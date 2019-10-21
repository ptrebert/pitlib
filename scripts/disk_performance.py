#!/usr/bin/env python3
# coding=utf-8

__author__ = "Peter Ebert"
__copyright__ = "Copyright (C) 2019 Peter Ebert"
__license__ = "MIT"

import os as os
import sys as sys
import traceback as trb
import argparse as argp
import json as json
import datetime as dt
import socket as skt
import random as rand
import string as string
import logging as logging
import logging.config as logconf

import numpy as np
import numpy.random as rng


logger = logging.getLogger()


BITS_PER_GB = 1024 * 1024 * 1024 * 8


def parse_command_line():
    """
    :return:
    """
    script_full_path = os.path.realpath(__file__)
    script_dir = os.path.dirname(script_full_path)
    log_config_default_path = os.path.join(script_dir, 'configs', 'log_config.json')
    if not os.path.isfile(log_config_default_path):
        script_root = os.path.split(script_dir)[0]
        log_config_default_path = os.path.join(script_root, 'configs', 'log_config.json')
        if not os.path.isfile(log_config_default_path):
            log_config_default_path = ''

    parser = argp.ArgumentParser(add_help=True, allow_abbrev=False)
    parser.add_argument('--debug', '-d', action='store_true', dest='debug',
                        help='Print progress messages (by default: to stderr).')

    parser.add_argument('--use-logger', '-ul', type=str,
                        default='default', dest='use_logger',
                        help='Name of logger to use (default).')
    parser.add_argument('--log-config', '-lc', type=str,
                        default=log_config_default_path, dest='log_config',
                        help='Full path to JSON file containing '
                             'configuration parameters for the '
                             'loggers to use. A logger named "debug" '
                             'must be present in the configuration file.')
    parser.add_argument('--output-folder', '-of', type=str, required=True, dest='outdir',
                        help='Path to store temp data.')
    parser.add_argument('--repeat', '-r', type=int, default=10, dest='repeat',
                        help='Repeat measurements this many times')
    parser.add_argument('--data-size', '-ds', type=int, default=4, dest='datasize',
                        help='Size of the test data file in GiB.')

    args = parser.parse_args()
    return args


def init_logger(cli_args):
    """
    :param cli_args:
    :return:
    """
    if not os.path.isfile(cli_args.log_config):
        return
    with open(cli_args.log_config, 'r') as log_config:
        config = json.load(log_config)
    if 'debug' not in config['loggers']:
        raise ValueError('Logger named "debug" must be present '
                         'in log config JSON: {}'.format(cli_args.logconfig))
    logconf.dictConfig(config)
    global logger
    if cli_args.debug:
        logger = logging.getLogger('debug')
    else:
        logger = logging.getLogger(cli_args.use_logger)
    logger.debug('Logger initialized')
    return


def main():
    """
    :return:
    """
    args = parse_command_line()
    init_logger(args)
    logger.debug('Starting performance test')
    os.makedirs(args.outdir, exist_ok=True)
    hostname = skt.gethostname()
    rand_string = ''.join(rand.sample(string.ascii_lowercase, 8))
    file_path = os.path.join(args.outdir, 'tmp_io-perf_{}_{}.npy'.format(hostname, rand_string))

    logger.info('Running on host: {}'.format(hostname))
    logger.info('Writing temp data to file: {}'.format(file_path))
    logger.info('Repeating measurements {} times'.format(args.repeat))

    timings = []
    speeds = []

    # assuming numpy default float64 dtype
    num_floats = int(args.datasize * BITS_PER_GB / 64)
    logger.debug('Generating {} random floats per iteration'.format(num_floats))

    for idx in range(args.repeat):
        rand_data = rng.random(num_floats)
        data_size_bytes = rand_data.nbytes
        data_size_mbytes = data_size_bytes / 1024 / 1024
        logger.debug(
            'Iteration {}: random data of size {} B (~ {} MB) generated'.format(
                idx + 1, data_size_bytes, data_size_mbytes
            )
        )
        logger.debug('Writing data...')
        with open(file_path, 'wb') as dump:
            start = dt.datetime.now()
            np.save(dump, rand_data, allow_pickle=False)
            end = dt.datetime.now()

        diff_in_sec = (end - start).total_seconds()
        mb_per_sec = data_size_mbytes / diff_in_sec
        timings.append(diff_in_sec)
        speeds.append(mb_per_sec)

        os.unlink(file_path)
        logger.debug('Iter complete')

    timings = np.array(timings, dtype=np.float64)
    timings = timings.round(2)
    speeds = np.array(speeds, dtype=np.float64)
    speeds = speeds.round(2)

    logger.info('Timings in seconds between I/O start and end')
    logger.info('Min.: {} s'.format(timings.min()))
    logger.info('Avg.: {} s'.format(timings.mean()))
    logger.info('Median: {} s'.format(np.median(timings)))
    logger.info('Max.: {} s'.format(timings.max()))
    logger.info('======================')
    logger.info('Write speed in MB/sec')
    logger.info('Min.: {} MB/s'.format(speeds.min()))
    logger.info('Avg.: {} MB/s'.format(speeds.mean()))
    logger.info('Median: {} MB/s'.format(np.median(speeds)))
    logger.info('Max.: {} MB/s'.format(speeds.max()))

    return


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        trb.print_exc(file=sys.stderr)
        sys.stderr.write('\nError: {}\n'.format(str(err)))
        sys.exit(1)
    else:
        sys.exit(0)
