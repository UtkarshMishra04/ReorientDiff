# ReorientDiff: Diffusion Model based Reorientation for Object Manipulation

Utkarsh A. Mishra and Yongxin Chen

Paper: [https://arxiv.org/abs/2303.12700](https://arxiv.org/abs/2303.12700)

## Skeleton Code for the paper

This repository contains the skeleton code for the paper [ReorientDiff: Diffusion Model based Reorientation for Object Manipulation](https://arxiv.org/abs/2303.12700).

The code builds on the works of Song et al. 2021, Janner et al. 2022 and Wada et al. 2022.

Mixed-Diffusion combines both the classifier-free and classifier-based guidance. This type of a formulation heps to estimate multi-modal distributions as a combination of a distribution conditioned on the structure of the task and distribution conditioned on the heuristics of solving the task.

This code is in further use for the next paper and a complete version will be released soon.

## Installation

### Requirements

Recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

## Citation

If you find this work useful in your research, please consider citing:

```
@article{mishra2023reorientdiff,
      title={ReorientDiff: Diffusion Model based Reorientation for Object Manipulation}, 
      author={Utkarsh A. Mishra and Yongxin Chen},
      year={2023},
      eprint={2303.12700},
      archivePrefix={arXiv}
}
```