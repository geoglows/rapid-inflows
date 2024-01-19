import argparse
import datetime
import logging
import sys

from .inflow import create_inflow_file


def main():
    parser = argparse.ArgumentParser(description='Create inflow file for LSM files and input directory.')

    parser.add_argument('--lsmdata', type=str,
                        help='Path to directory of LSM NC files, a single NC file, or a glob pattern for NC files',
                        required=True)
    parser.add_argument('--inputdir', type=str,
                        help='Input directory path for a specific VPU which contains weight tables',
                        required=True)
    parser.add_argument('--inflowdir', type=str,
                        help='Inflow directory path for a specific VPU where m3 runoff inflows NC files are saved',
                        required=True)
    parser.add_argument('--timestep', type=int, default=3,
                        help='Desired time step in hours. Default is 3 hours',
                        required=False)
    parser.add_argument('--cumulative', action='store_true', default=False,
                        help='A boolean flag to mark if the runoff is cumulative. Inflows should be incremental',
                        required=False)
    parser.add_argument('--file_label', type=str, default=None,
                        help='A string to include in the inflow file name. Default is None',
                        required=False)

    args = parser.parse_args()

    gen(parser, args)
    return


def gen(parser, args):
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Access the parsed argument
    lsm_data = args.lsmdata
    input_dir = args.inputdir
    inflow_dir = args.inflowdir
    timestep = datetime.timedelta(hours=args.timestep)
    cumulative = args.cumulative
    file_label = args.file_label

    if not all([lsm_data, input_dir, inflow_dir]):
        # print usage
        parser.print_usage()
        return

    # Create the inflow file for each LSM file
    create_inflow_file(lsm_data, input_dir, inflow_dir, timestep=timestep, cumulative=cumulative, file_label=file_label)
    return
