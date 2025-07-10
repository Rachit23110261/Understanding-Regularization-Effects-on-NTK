# Regularization Effects on NTK Spectrum and Generalization

Understanding how regularization techniques affect the Neural Tangent Kernel (NTK) matrix and its eigenvalue spectrum, with theoretical insights and experimental visualizations.

## Motivation

The Neural Tangent Kernel (NTK) provides a powerful lens to understand generalization in overparameterized neural networks. This project investigates how regularization techniques such as L2 penalty, L1 penalty and early stopping influence the NTK spectrum and, ultimately, model generalization behavior.

## Background: What is NTK?

In the infinite-width limit, neural networks trained with gradient descent exhibit linearized training dynamics around their initialization. The Neural Tangent Kernel (NTK) is a data-dependent kernel that captures this behavior. Basically it is the change in Neural network function with respect to small change in parameter.

Mathematically, for a neural network function 


<img width="300" height="168" alt="image" src="https://github.com/user-attachments/assets/61d4f2ea-b404-4c28-a4a4-eba7d40d13b5" />



## 🧪 Empirical Neural Tangent Kernel (NTK) Matrix

Given a neural network $f(\mathbf{x}, \theta)$ with parameters $\theta \in \mathbb{R}^P$, the **Neural Tangent Kernel** between two inputs $\mathbf{x}$ and $\mathbf{x}'$ is defined as:

$$
\Theta(\mathbf{x}, \mathbf{x}') = \nabla_\theta f(\mathbf{x}, \theta)^\top \nabla_\theta f(\mathbf{x}', \theta)
$$

This captures how changes in parameters affect predictions at different input points.

When evaluated on a dataset $\{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n\}$, the **empirical NTK matrix** is the $n \times n$ matrix:

$$
\Theta_{ij} = \Theta(\mathbf{x}_i, \mathbf{x}_j)
= \nabla_\theta f(\mathbf{x}_i, \theta)^\top \nabla_\theta f(\mathbf{x}_j, \theta)
$$

This matrix measures the pairwise similarity of data points in the **parameter gradient space**.

---
## Spectral Analysis

Let $\Theta \in \mathbb{R}^{n \times n}$ be the empirical NTK matrix. Its eigenvalues $\lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_n$ encode the **learning dynamics**:

- Directions with larger eigenvalues are **learned faster**.
- Smaller eigenvalue modes take longer to fit or are **suppressed by regularization**.
- The shape of the spectrum relates to **generalization performance**.

---

## Why This Matters

In the **infinite-width limit**, training with gradient descent behaves like **kernel regression** with the NTK:

$$
\frac{d}{dt} f(\mathbf{x}, t) = - \sum_{j=1}^n \Theta(\mathbf{x}, \mathbf{x}_j) \frac{\partial \mathcal{L}}{\partial f(\mathbf{x}_j)}
$$

This shows that the NTK determines **how the network learns** during training.
