This README provides a comprehensive overview of the MATLAB implementation and experimental scripts for the **Tube Loss** function, as presented in the research paper:

> *Anand, P., Bandyopadhyay, T., & Chandra, S. "Tube Loss: A Novel Approach for Prediction Interval Estimation and probabilistic forecasting."*

## Introduction

This folder contains the source code for implementing Tube Loss using kernel-based methods (e.g., Support Vector Regression frameworks) in MATLAB. While the ANN implementation (found in the parent directory) focuses on deep learning architectures, these scripts provide a robust framework for evaluating Tube Loss with various kernel functions (Linear, RBF, etc.) and specific optimization routines.

The experiments demonstrate the efficiency of Tube Loss in constructing Prediction Intervals (PIs) that satisfy target coverage probabilities while minimizing interval width, especially in the presence of skewed noise.

---

## Directory Structure & File Descriptions

### Core Functions
* **`cce_withR_m1.m` / `cce_withR_m2.m`**: The primary implementation of the Tube Loss optimization algorithm. These functions calculate the decision boundaries ($f_U$ and $f_L$) by minimizing the empirical risk. The scripts handle the iterative updates of weights and biases using a gradient-based approach.
* **`svkernel.m`**: A utility function providing various kernel implementations, including Linear, Polynomial, and Radial Basis Function (RBF). This allows the Tube Loss to be applied to non-linear datasets by mapping them into high-dimensional feature spaces.

### Synthetic Experiments
* **`Exp1.m` / `Test1.m`**: Focused on symmetric noise distributions.
    * **Data**: $y = \frac{\sin(x)}{x} + \epsilon$, where $\epsilon \sim \mathcal{N}(0, \sigma^2)$.
    * **Objective**: Demonstrates that Tube Loss asymptotically converges to the true symmetric PI bounds.
* **`Exp2.m` / `Test2.m`**: Focused on asymmetric/skewed noise distributions.
    * **Data**: $y = \frac{\sin(x)}{x} + \epsilon$, where $\epsilon \sim \chi^2(df=3)$.
    * **Objective**: Illustrates the significance of the **$r$ parameter**. By adjusting $r$, the "tube" shifts to capture the high-density regions of the Chi-square distribution, resulting in narrower intervals for the same coverage level compared to standard methods.

### Real-World Benchmarking
* **`servo8.m`**: Implementation of Tube Loss on the **UCI Servo Dataset**.
    * This script demonstrates the practical application of the loss function on real-world engineering data, showcasing its ability to provide reliable PIs in non-synthetic environments.

---

## Technical Specifications

### Key Parameters
The following parameters are utilized across the `.m` scripts to control the behavior of the Tube Loss:
* `para.r`: The bias parameter $[0, 1]$ that controls the vertical shift of the interval to accommodate skewed data.
* `para.delta`: The insensitive zone parameter, defining the "tube" width where no penalty is applied.
* `para.lambda`: Regularization parameter to prevent overfitting.
* `para.eta` & `para.eps1`: Learning rate and decay rate for the optimization solver.
* `para.kernel`: Integer identifier for the kernel type (1 for Linear, 2 for RBF).

### Optimization Approach
The scripts utilize a custom gradient descent routine to solve the dual or primal problem (depending on the specific script). The solver is designed to handle the non-differentiable points of the Tube Loss function, ensuring stable convergence to the optimal PI boundaries.

---

## How to Run

1.  **Environment**: Ensure MATLAB is installed (tested on R2021b and later).
2.  **Running Synthetic Tests**:
    * Execute `Exp1.m` to observe PI estimation on Gaussian noise.
    * Execute `Exp2.m` to observe the $r$-effect on Chi-square noise.
3.  **Real-World Data**:
    * Ensure `servo.mat` is in the path.
    * Run `servo8.m` to see performance metrics (Coverage and Length) on the Servo dataset.
4.  **Customization**: Modify the `para` struct within any `Exp` or `Test` script to experiment with different confidence levels (`tau`) or kernel hyperparameters.

## Performance Metrics
The experiments evaluate the model based on:
1.  **PICP (Prediction Interval Coverage Probability)**: The percentage of test points falling within the predicted bounds.
2.  **PINAW (Prediction Interval Normalized Average Width)**: The average width of the intervals, normalized to the range of the underlying data.
3.  **Loss Value**: The final value of the objective function post-convergence.
