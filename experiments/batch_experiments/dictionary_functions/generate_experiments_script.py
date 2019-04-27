
from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)

import os
import sys
import argparse

NUM_TRIALS = 3
WEAK_WEIGHT = [0.01]
WEAK_FUNCTIONS = ['linear', 'knn', 'keyword']
NUM_EPOCHS = 5

def generate_experiment_commands() -> List[str]:
    commands: List[str] = []
    for trial in range(NUM_TRIALS):
        for weak_weight in WEAK_WEIGHT:
            for weak_function in WEAK_FUNCTIONS:
                model_name = f'{weak_weight}/{weak_function}/trial_{trial}'
                command = f'python dpd/allennlp_active_train.py --weak_function {weak_function} --num_epochs {NUM_EPOCHS} --weak_weight {weak_weight} --model_name {model_name} --use_weak --cuda --use_weak_fine_tune'
                commands.append(command)
    return commands

def get_bash_header() -> List[str]:
    return ['#!/bin/bash','set -e'] 

def main():
    parser = argparse.ArgumentParser(description='Generates a bash script for running many experiments')
    bash_commands = get_bash_header() + generate_experiment_commands()
    bash_script = os.path.join(os.path.dirname(__file__), 'weak_fine_tune_active_model_experiments.sh')
    with open(bash_script, 'w') as f:
        for line in bash_commands:
            f.writelines(f'{line}\n')

if __name__ == "__main__":
    main()