#!/bin/bash

#python run_experiment.py --normalized True --model Ridge --pad_phase_on True

python impute_data.py
python impute_data.py --pad_phase_on
python impute_data.py --normalized
python impute_data.py --pad_phase_on --normalized