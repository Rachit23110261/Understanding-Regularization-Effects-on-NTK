# Regularization Effects on NTK Spectrum and Generalization

Understanding how regularization techniques affect the Neural Tangent Kernel (NTK) matrix and its eigenvalue spectrum, with theoretical insights and experimental visualizations.

---

## Motivation

The Neural Tangent Kernel (NTK) provides a powerful lens to understand generalization in overparameterized neural networks. This project investigates how regularization techniques such as **L2 penalty**, **L1 penalty**, and **early stopping** influence the NTK spectrum and, ultimately, the model's generalization behavior.

---

## Background: What is NTK?

In the infinite-width limit, neural networks trained with gradient descent exhibit **linearized training dynamics** around their initialization. The NTK is a data-dependent kernel that captures this behavior — it measures how a network function changes with small perturbations in parameters.

Mathematically, for a neural network function $\( f(\mathbf{x}, \theta) \)$, the NTK is defined as:

<div align="center">
  <img width="300" alt="NTK Equation" src="https://github.com/user-attachments/assets/61d4f2ea-b404-4c28-a4a4-eba7d40d13b5" />
</div>

---

## Empirical Neural Tangent Kernel (NTK) Matrix

Given a neural network $\( f(\mathbf{x}, \theta) \)$ with parameters $\( \theta \in \mathbb{R}^P \)$, the **NTK between two inputs** $\( \mathbf{x} \)$ and $\( \mathbf{x}' \)$ is:

$$
\Theta(\mathbf{x}, \mathbf{x}') = \nabla_\theta f(\mathbf{x}, \theta)^\top \nabla_\theta f(\mathbf{x}', \theta)
$$

This captures how changes in parameters affect predictions at different input points.

For a dataset $\( \{\mathbf{x}_1, \dots, \mathbf{x}_n\} \)$, the **empirical NTK matrix** $\( \Theta \in \mathbb{R}^{n \times n} \)$ is:

$$
\Theta_{ij} = \nabla_\theta f(\mathbf{x}_i, \theta)^\top \nabla_\theta f(\mathbf{x}_j, \theta)
$$

This matrix measures **pairwise similarity in the parameter gradient space**.

---

## Spectral Analysis of NTK

Let $\( \Theta \in \mathbb{R}^{n \times n} \)$ be the empirical NTK matrix with eigenvalues $\( \lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_n \)$. These eigenvalues encode learning dynamics:

- **Larger eigenvalues** → Directions that are learned faster.
- **Smaller eigenvalues** → Slower modes, more suppressed by regularization.
- The **spectrum shape** is linked to generalization performance.

The NTK also governs training dynamics:

$$
\frac{d}{dt} f(\mathbf{x}, t) = - \sum_{j=1}^n \Theta(\mathbf{x}, \mathbf{x}_j) \frac{\partial \mathcal{L}}{\partial f(\mathbf{x}_j)}
$$

---

## Experimental Results and Explanations

### Effect of L2 Regularization on NTK Eigenvalues

<div align="center">
  <img width="1437" alt="L2 Spectrum" src="https://github.com/user-attachments/assets/c8ffe4aa-2cef-4ac4-811c-f2001be9c138" />
</div>

- Increasing the L2 regularization leads to a **steady drop in the largest eigenvalue**.
- This indicates **reduced model expressiveness**.
- The network focuses more on minimizing parameter norm than optimizing loss.

---

### NTK Spectrum Evolution During Training

<div align="center">
  <img width="1149" alt="NTK vs Epochs" src="https://github.com/user-attachments/assets/91cf5b7b-9db5-412f-b81e-de87b0ed16d4" />
</div>

- NTK eigenvalue evolution reflects learning speed and overfitting behavior.
- Early in training: large changes in parameters (random weights).
- Mid-training: stabilized learning (small parameter updates).
- Late training: risk of **overfitting**, causing NTK to increase again.

---

### Maximum Eigenvalue and Parameter Sensitivity

Key interpretations:

1. NTK captures sensitivity of outputs to small parameter changes.
2. Eigenvectors: directions of highest/lowest sensitivity.
3. **Max eigenvalue** → fastest change in output → sharp learning direction.

- Early stage: Large parameter updates.
- Later: Small updates, risk of overfitting increases eigenvalue again.

---

### Effect of Network Width on Largest NTK Eigenvalue

<div align="center">
  <img width="1188" alt="Width vs NTK" src="https://github.com/user-attachments/assets/9d57a08b-f2c1-4158-985c-fee6dfdcba8a" />
</div>

- Wider networks typically have higher maximum eigenvalues.
- Indicates higher capacity to capture variance and learn quickly.

---

### Training Data Size vs Maximum NTK Eigenvalue

<div align="center">
  <img width="1052" alt="Training Size NTK" src="https://github.com/user-attachments/assets/b28c7c7e-112c-4394-8ef9-a555107d13eb" />
</div>

- More training data → more data per epoch → larger parameter updates.
- Leads to higher max eigenvalue.

---

### Effect of L2 Regularization on Overfitting

<div align="center">
  <img width="1329" alt="L2 Regularization NTK" src="https://github.com/user-attachments/assets/6a1a7f0f-2388-46b7-a5e3-4697161fc50d" />
</div>

- L2 regularization **reduces overfitting** by penalizing large weights.
- Sharp decline in NTK eigenvalue beyond optimal epoch.
- Higher L2 → quicker suppression of high-capacity directions.

---

### Intuitive View: NTK Collinearity and Trace

- NTK matrix size: $\( n \times n \)$
- **Trace of NTK** is the sum of eigenvalues:

$$
\text{Tr}(\Theta) = \sum_{i=1}^n \lambda_i
$$

- The trace scales approximately linearly with dataset size $\( n \)$.
- Most of the trace is concentrated in **top few eigenvalues**.
- Therefore, the **maximum eigenvalue** increases approximately linearly with dataset size.

<div align="center">
  <img width="1145" alt="Trace NTK" src="https://github.com/user-attachments/assets/7f3b116d-318b-4e1b-bea4-44b597a366b8" />
</div>

<div align="center">
  <img width="1176" alt="NTK Growth 1" src="https://github.com/user-attachments/assets/e57b24fc-59cc-4bef-900f-30f94e522cb0" />
</div>

<div align="center">
  <img width="1329" alt="NTK Growth 2" src="https://github.com/user-attachments/assets/74689559-5d6f-4fc3-afbf-5b47c67a626b" />
</div>

---

## Conclusion

- The **maximum eigenvalue** of the NTK serves as a key indicator of how fast the network learns.
- **Regularization techniques** (L1, L2, early stopping) suppress specific modes in the NTK spectrum, directly affecting generalization.
- **Spectral analysis** of the NTK helps visualize and explain the behavior of deep networks under regularization.

