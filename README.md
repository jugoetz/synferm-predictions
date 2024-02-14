# synferm-predictions

## Installation

Notes:
- You can try installing with `conda`, but the `mamba` solver worked more reliably for us
- The `environment.yaml` file is written for a workstation with Nvidia GPU.
   Remove the CUDA dependencies to install for use on CPU.

```bash
mamba env create -f environment.yaml
```

## Training models
The `run.py` script serves as an entrypoint for training models.
It is configured with a set of command line arguments,
including the path to a configuration file with model hyperparameters.
See `config/config.yaml` for an example configuration file.

To see the full list of command line arguments, run:
```bash
python run.py train --help
```

## Predicting using trained models
The `inference.py` script serves as an entrypoint for predicting reaction outcome.
It expects a CSV file with three columns: `initiator`, `monomer`, `terminator`.
See `config/config.yaml` for an example configuration file.

Call it like:
```bash
python inference.py -i example_reactants.csv -o out.csv
```
or use `python inference.py --help` for more information.
