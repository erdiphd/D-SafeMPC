# D-SafeMPC: Diffusion-Driven Safe Model Predictive Control with Discrete-Time Control Barrier Functions

This repository contains the official implementation of our paper:

> **D-SafeMPC: Diffusion-Driven Safe Model Predictive Control with Discrete-Time Control Barrier Functions**

We build upon the temporal U-Net implementation from [Diffuser](https://github.com/jannerm/diffuser) and use the simulation environment from [D3IL](https://github.com/ALRhub/d3il).

---

## Overview

A key limitation of diffusion models in robotic planning is their inability to inherently enforce safety or dynamical constraints, often resulting in physically infeasible or unsafe outputs. Hybrid approaches that employ Model Predictive Control (MPC) can be unstable, as poor trajectory initializations from the diffusion model prevent the MPC from converging to a safe and feasible solution.

**D-SafeMPC** addresses these challenges by enhancing the interaction between diffusion and control:

- **Guided Reverse Diffusion:** The reverse diffusion process is steered using gradients from discrete-time Control Barrier Functions (CBFs) and Control Lyapunov Functions (CLFs), pushing trajectory generation toward safe, goal-directed regions.
- **Iterative MPC Projection:** At each denoising step, an MPC refines the trajectory by projecting it onto a safe and dynamically feasible set, enforcing hard safety and dynamic constraints.
- **Reliable Warm Starts:** The CBF/CLF guidance provides well-initialized trajectories for MPC, enabling more stable and efficient convergence within the allocated computational budget.

This iterative cycle of guided denoising and MPC projection continues until the final timestep, at which point the control action from the converged trajectory is applied to advance the robot to the next state.

## Key Contributions

1. A novel framework leveraging **discrete-time CBFs and CLFs** to guide the reverse diffusion process, steering trajectory generation toward safer, goal-oriented regions and improving MPC initialization.
2. An **MPC-based projection** that enforces strict safety and dynamic constraints on the probabilistically safe trajectories, ensuring the resulting trajectory remains within a defined safe set.
3. **Empirical validation** on a Franka robot manipulator across four scenarios (one static-obstacle and three dynamic-obstacle settings), including sim-to-real experiments on a physical Franka robot, demonstrating superior safety, task success rates, and computational efficiency over state-of-the-art baselines.

---

## Installation

Clone the repository and set up the environment:

```bash
conda create -n dsafempc python=3.10
conda activate dsafempc
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

You also need to install [D3IL](https://github.com/ALRhub/d3il) for the simulation environment.

## Training

To train the diffusion policy, run:

```bash
python scripts/train.py
```

You can also visualize the training data and constraints:

```bash
python scripts/visualize_data_constraints.py
```

## Evaluation

To evaluate D-SafeMPC and reproduce the results reported in the paper, run:

```bash
python scripts/eval.py
python scripts/load_results.py
```

---

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{dsafempc2025,
  title     = {D-SafeMPC: Diffusion-Driven Safe Model Predictive Control with Discrete-Time Control Barrier Functions},
  author    = {},
  year      = {2025}
}
```

## Acknowledgments

This project builds upon the following works:

- [Diffuser](https://github.com/jannerm/diffuser) — Temporal U-Net for diffusion-based planning
- [D3IL](https://github.com/ALRhub/d3il) — Simulation environment
