## [ICASSP 2026] REGION GROWING PHYSICS-INFORMED NEURAL NETWORK FOR WIND FIELD RECONSTRUCTION FROM SPARSE DATA

> **Authors:**

Lei Liu, Ziqian Hu, Hongwei Zhao$^\dagger$, Jiahui Huang, Tengyuan Liu, Hong Wang, Bin Li

This repo contains the code and data from our paper in ICASSP 2026

## 1. Abstract

Reconstructing wind fields from observational data is crucial for applications in meteorology, weather forecasting, and renewable energy systems. However, existing Physics-Informed Neural Networks (PINNs) struggle under sparse observations, as insufficient data provides inadequate guidance for accurate reconstruction. To address this challenge, we propose a Region-Growing PINN (RG-PINN) for wind field reconstruction governed by the Navier–Stokes (NS) equations. Instead of training over the entire domain from the beginning, our method starts in data-dense subregions and gradually expands the training domain. This progressive strategy enables the model to first stabilize in well-observed areas under strong data-physical consistency, before generalizing to sparser regions. To further enhance learning, we design a dynamic point-wise weighting scheme that adaptively balances the loss contributions based on local learning difficulty. Additionally, a space-time averaged PDE loss is introduced to reduce the influence of noisy observations and non-ideal physical conditions. Experiments on real-world wind field datasets show that RG-PINN achieves up to a 21.3% reduction in RMSE compared to the current state-of-the-art (SOTA) PINN method. This work presents a novel training paradigm for physics-informed learning under sparse data conditions.

## 2. Requirements

The version of python is 3.10.13 .
The version of torch is 1.13.1 .

```bash
# Core deep learning framework
torch>=1.9.0
torchvision>=0.10.0

# Scientific computing
numpy>=1.21.0
scipy>=1.7.0

# Data handling and visualization
h5py>=3.1.0
matplotlib>=3.4.0
pandas>=1.3.0

# Progress bars and utilities
tqdm>=4.62.0
```

## 3. Datasets

The URLs of used datasets are as follows:

CERRA dataset: https://climate.copernicus.eu/copernicus-regional-reanalysis-europe-cerra

BLexp dataset: https://deepblue.lib.umich.edu/data/concern/data_sets/

The preprocessed datasets are provided.

## 4. Usage

- an example for train and evaluate a new model：

```bash
python /BLExp/main.py
```

```
python /CERRA/main2.py
```

## 5. Acknowledgments



## 6. Citation

If you find our work useful in your research, please consider citing:

```latex
@inproceedings{rgpinn,
  title={REGION GROWING PHYSICS-INFORMED NEURAL NETWORK FOR WIND FIELD RECONSTRUCTION FROM SPARSE DATA},
  author={Lei Liu, Ziqian Hu, Hongwei Zhao, Jiahui Huang, Tengyuan Liu, Hong Wang, Bin Li},
  booktitle={IEEE International Conference on Acoustics, Speech, and Signal Processing},
  year={2025}
}
```

If you have any problems, contact me via liulei13@ustc.edu.cn.


