#!/bin/bash

#SBATCH --job-name="pre_processing_exp"

#SBATCH --account=3144366

#SBATCH --partition=defq

#SBATCH --output=out/%x_%j.out # %x gives job name and %j gives job id

#SBATCH --error=err/%x_%j.err

#SBATCH --mail-type=END

#SBATCH --mail-user=dario.filatrella@studbocconi.it

python pre_processing_experiments.py





