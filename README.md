# HCVA-plus-plus

This repository contains the complete code for the paper "HCVA: Hierarchical Consensus Value Aggregation", by Joseph Trevorrow, Emanuele Ratti, and Nirav Ajmeri.

## Setup

The code requires both Python and Julia, and has been tested on an Ubuntu linux machine, and an aarch HPC cluster. Your results following this set up may vary if you are using Windows or MacOS. Please refer to the original package/language websites if trouble ensues! 

To set up Python:
```shell
conda create -n hcva --python=3.12.13
conda activate hcva
python -m pip install -r requirements.txt
```

To set up Julia:
```shell
curl -fsSL https://install.julialang.org | sh
```

## Running Experiments
All run scripts can be found in the `shellscripts` folder. Scripts starting "HPC" are set up to be used with a SLURM job management system. The other scripts are as follows:

- `duplicator.sh` duplicates the "HPC" scripts so that multiple jobs can be run simultaneously.
- `run_hpc_sh.sh` is a simple for loop that `sbatch`s all "HPC" scripts.
- `synth_data_gen.sh` generates the synthetic data. Note that there are **two seeds**. One seed is used for synthetic data experiments, the other seed is for the time complexity experiments. This is clearly commented at the start of `synth_data_generator.py`.
- `time_complexity_exp.sh` runs the time complexity experiments.

### HPC (or synthetic data) experiments
By default, HPC experiments are set up to run multithreaded. Single threaded experiments can be run by searching in `runner.py` for the comment `# non-parallel`.

There are three run scripts:
1. `synth_data_generator.py` runs the synthetic data generation, and all functions that are used to generate synthetic data can be found here
2. `runner.py` is the main run script for ESS and synthetic data experiments. All multithreading, preprocessing of input data, and saving results to a file can be found here. Computation code can be found in the `lp_regression` folder.
3. `time_complexity_runner.py` is the exact same as `runner.py`, but saves data differently, only considering the time taken for computation. 



## Input Data, Results, and Data Analysis

### Input Data

- Each experiment has separate PriP and PVS dirs. Each dir has 500 subdirs, each representing an individual "run". Each experiment (say, randoms), runs 500 times with different input data. If an experiment has multiple society characteristics (i.e. the group size increasing), then each "run" will be a complete replication.
- You can find input_data in `value_systems`, by generating the input data by running `time_complexity_runner.py`

### Results

Results are split by type. All directories containing results are listed `results-`.

### Data Analysis

Data analysis results can be found in directories beginning `plots-`. Data analysis code can be found in `data_analysis`.

`data_analysis` contains 