#!/bin/bash

#SBATCH --job-name="gurobi_test"

#SBATCH --account=3144366

#SBATCH --partition=defq

#SBATCH --output=out/%x_%j.out # %x gives job name and %j gives job id

#SBATCH --error=err/%x_%j.err

#SBATCH --mail-type=END

#SBATCH --mail-user=dario.filatrella@studbocconi.it

python gurobi_test.py





