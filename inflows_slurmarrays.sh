#!/bin/bash --login

conda activate inflows

ls /home/rchales/compute/input/* | \
slurm-auto-array --mem-per-cpu 2048M --time 00:25:00 --mail-type=BEGIN --mail-type=END --mail-type=FAIL --mail-user=rchales@byu.edu --ntasks 4 \
-- python /home/rchales/rapid-inflows/inflows_fast.py --lsmdir /home/rchales/compute/era5_daily/$1 --inflowdir /home/rchales/compute/inflows --inputdir