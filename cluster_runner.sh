#!/bin/bash
sbatch -n 4 --mem-per-cpu=2048 --time=24:00:00 --wrap="python -u checker_client.py --results-dir ."
