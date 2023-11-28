#!/bin/bash

echo "Starting project environment setup"

# These commands need to be run manually unfortunately
# Simply copy and run this command
module load Python/3.10.8-GCCcore-12.2.0 ; export PYTHONUSERBASE=$HOME/.usr/local/python/3.10.8 ; mkdir -p $PYTHONUSERBASE ; export PATH=$PATH:$HOME/.usr/local/python/3.10.8/bin

# These commands work fine
python -m pip install --user Flask
python -m pip install --user pandas
python -m pip install --user transformers
python -m pip install --user datasets
python -m pip install --user codecarbon
python -m pip install --user torch
python -m pip install --user transformers[torch]
python -m pip install --user peft

echo "Finished setup"
