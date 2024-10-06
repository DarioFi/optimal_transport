#!/bin/bash

#SBATCH --job-name="gmmx_n=5"

#SBATCH --account=3144366

#SBATCH --partition=defq

#SBATCH --output=out/%x_%j.out # %x gives job name and %j gives job id

#SBATCH --error=err/%x_%j.err

#SBATCH --mail-type=END

#SBATCH --mail-user=dario.filatrella@studbocconi.it

python experiment.py





