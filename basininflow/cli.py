import argparse
import datetime
import logging
import os
import sys

from .inflow import create_inflow_file


def main():
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='Create inflow file for LSM files and input directory.')

    # Define the command-line argument
    parser.add_argument('--lsmdir', type=str,
                        help='Directory of LSM files')
    parser.add_argument('--inputdir', type=str,
                        help='Input directory path for a specific VPU which contains weight tables')
    parser.add_argument('--inflowdir', type=str,
                        help='Inflows directory which contains VPU subdirectories')
    parser.add_argument('--timestep', type=int, default=3,
                        help='Desired time step in hours. Default is 3 hours')
    parser.add_argument('--cumulative', action='store_true', default=False,
                        help='A boolean flag to mark if the runoff is cumulative. Inflows should be incremental')
    args = parser.parse_args()

    # Access the parsed argument
    lsm_data = args.lsmdir
    input_dir = args.inputdir
    inflows_dir = args.inflowdir
    timestep = datetime.timedelta(hours=args.timestep)
    cumulative = args.cumulative

    if not all([lsm_data, input_dir, inflows_dir]):
        raise ValueError('Missing required arguments --lsmdir, --inputdir, --inflowdir')

    # check what kind of lsm data was given
    if os.path.isdir(lsm_data):
        lsm_data = os.path.join(lsm_data, '*.nc*')
    elif os.path.isfile(lsm_data):
        ...  # this is correct
    elif not os.path.exists(lsm_data) and '*' not in lsm_data:
        raise FileNotFoundError(f'{lsm_data} does not exist and is not a glob pattern')

    # Create the inflow file for each LSM file
    create_inflow_file(lsm_data,
                       input_dir,
                       inflows_dir,
                       cumulative=cumulative,
                       timestep=timestep)
