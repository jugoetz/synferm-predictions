# synferm-predictions

## Installation

Notes:
- You can try installing with `conda`, but the `mamba` solver worked more reliably for us
- The `environment.yaml` file is written for a workstation with Nvidia GPU.
   Remove the CUDA dependencies to install for use on CPU.

```bash
mamba env create -f environment.yaml
```
