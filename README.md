# ML Training and Inference for Synthetic Fermentation

<a href="https://github.com/jugoetz/synferm-predictions/blob/main/LICENSE"><img alt="License: MIT" src="https://black.readthedocs.io/en/stable/_static/license.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

For code used to collect experimental data, see [this repository](https://github.com/jugoetz/library-generation).

## Installation

```bash
conda env create -f environment.yaml
```
or (if you don't have a suitable GPU):
```bash
conda env create -f environment_cpuonly.yaml
```

_Notes:_
- The `environment.yaml` file is written for a workstation with Nvidia GPU.
   Use `environment_cpuonly.yaml` instead to run only on CPU.
   This will not install `CUDA` and will install the CPU-only versions of `pytorch` and `dgl`.
- Installing the `dgl` dependency through `conda` sometimes creates issues where some packages are "not found" despite existing in the specified channels.
   Instead, try installing `dgl` separately with `pip`:
   ```bash
   pip install dgl -f https://data.dgl.ai/wheels/repo.html
   ```
- On some systems with outdated libraries (such as university clusters) `dgl` wheels may not work,
   and you may need to build it from source.
   See the [DGL installation guide](https://docs.dgl.ai/install/index.html) for more information.
- There is an issue with `pytorch` and the `2024.1.x` version of the `mkl` dependency.
  If an `ImportError [...] undefined symbol: iJIT_NotifyEvent` occurs, downgrade with `conda install mkl=2024.0`

### Log in to WandB
We track training runs with [WandB](https://wandb.ai).
Before starting any training runs, you need to log into `WandB` by running
```bash
wandb login
```
then supply your API key.

## Training models
The `run.py` script serves as an entrypoint for training models.
It is configured with a set of command line arguments,
including the path to a configuration file with model hyperparameters.
See `config/config_example.yaml` for an example configuration file.

To see the full list of command line arguments, run:
```bash
python run.py train --help
```

## Predicting using trained models
The `inference.py` script serves as an entrypoint for predicting reaction outcome.
It expects a CSV file with three columns: `initiator`, `monomer`, `terminator`.
See `config/config_example.yaml` for an example configuration file.

Call it like:
```bash
python inference.py -i example_reactants.csv -o out.csv
```
or use `python inference.py --help` for more information.

## Development
We use [nbstripout](https://pypi.org/project/nbstripout/) to remove output from notebooks before committing to the repository.
Install with:
```bash
conda install -c conda-forge nbstripout # or pip install nbstripout
nbstripout --install  # configures git filters and attributes for this repo
```
We use [pre-commit](https://pre-commit.com/) hooks to ensure standardized code formatting.
Install with:
```bash
pre-commit install
```
