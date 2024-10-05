#!/bin/bash

#SBATCH --job-name="baron_test"

#SBATCH --account=3144366

#SBATCH --qos=debug
#SBATCH --partition=debug

#SBATCH --output=out/%x_%j.out # %x gives job name and %j gives job id

#SBATCH --error=err/%x_%j.err

#SBATCH --mail-type=END

#SBATCH --mail-user=dario.filatrella@studbocconi.it

python experiment.py





