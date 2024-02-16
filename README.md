# VHIP balancing in Python

Python prototyping code for [Biped Stabilization by Linear Feedback of the Variable-Height Inverted Pendulum Model](https://arxiv.org/abs/1909.07732).

## Usage

If you already have CVXPY, you can run this script directly:

```console
python main.py
```

## Installation

If you don't have all dependencies, you can get them all at once using Conda:

```
conda create -f environment.yaml
conda activate vhip_balancers
```

## See also

- [VHIP walking controller](https://github.com/stephane-caron/vhip_walking_controller): C++ version of the VHIP QP balancer that was validated on the HRP-4 humanoid robot.
