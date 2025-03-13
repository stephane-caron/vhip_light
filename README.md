# VHIP balancing in Python

Python code for [Biped Stabilization by Linear Feedback of the Variable-Height Inverted Pendulum Model](https://arxiv.org/abs/1909.07732).

The main script unrolls VHIP trajectories with different balance controllers and reproduces plots as in the paper:

![image](https://github.com/stephane-caron/vhip_light/assets/1189580/8a12ac28-3402-4b7a-92ea-9e46e18849b3)

This repository is a lighter implementation of the original script [vhip\_stabilization.py](https://github.com/stephane-caron/pymanoid/blob/2cbce7441f377a18119837dfeee51662654c6ce7/examples/vhip_stabilization.py) that was released with the paper.

## Installation

The recommended way is to use Conda:

```console
conda create -f environment.yaml
conda activate vhip_light
```

Alternatively, you can install all dependencies from PyPI:

```console
pip install cvxpy ipython matplotlib qpsolvers scipy
```

## Usage

```console
python main.py
```

## Citation

If you find this code or the paper useful, please consider citing it:

```bibtex
@inproceedings{caron2020icra,
  title = {Biped Stabilization by Linear Feedback of the Variable-Height Inverted Pendulum Model},
  author = {Caron, St{\'e}phane},
  booktitle = {IEEE International Conference on Robotics and Automation},
  url = {https://hal.archives-ouvertes.fr/hal-02289919},
  year = {2020},
  month = may,
  doi = {10.1109/ICRA40945.2020.9196715},
}
```

## See also

- [VHIP stabilization script](https://github.com/stephane-caron/pymanoid/blob/2cbce7441f377a18119837dfeee51662654c6ce7/examples/vhip_stabilization.py): original Python prototype as a single script, with dependencies on OpenRAVE and pymanoid (now deprectated).
- [VHIP walking controller](https://github.com/stephane-caron/vhip_walking_controller): C++ version of the VHIP QP balancer that was validated on the HRP-4 humanoid robot.
