This README provides a research-grade overview of the experimental files associated with the **Tube Loss** paper.

---

# Experiments for Tube Loss: A Novel Approach for Prediction Interval Estimation

This directory contains the source code and experimental setups for implementing and evaluating the **Tube Loss** function, as introduced in:
> *Anand, P., Bandyopadhyay, T., & Chandra, S. "Tube Loss: A Novel Approach for Prediction Interval Estimation and probabilistic forecasting."*

## Overview

The **Tube Loss** is a novel loss function designed for the simultaneous estimation of the upper and lower bounds of a Prediction Interval (PI). Unlike traditional methods that may require separate optimization for each bound or assume specific distributions, Tube Loss allows for a trade-off between coverage and width through a single optimization problem. 

A key feature of Tube Loss is the **$r$ parameter**, which allows users to shift the interval to capture denser regions of the conditional distribution, effectively sharpening the PI width when the response variable is skewed.

## File Descriptions

### 1. `Tube_loss_ANN.ipynb`
This notebook provides the core implementation of the Tube Loss within an Artificial Neural Network (ANN) framework.
* **Experimental Setup**: Uses a synthetic dataset generated from $y = \frac{\sin(x)}{x} + \epsilon$, where $X \in [-2\pi, 2\pi]$.
* **Noise Profiles**: Tests the model against symmetric noise distributions, including **Uniform** and **Normal**.
* **Key Functionality**: Defines the Keras-based implementation of the Tube Loss and demonstrates that the estimated PIs asymptotically converge to the true boundaries of the distribution at a specified confidence level.

### 2. `Tube_loss_ANN_r_effect.ipynb.ipynb`
This notebook focuses on the impact of the $r$ parameter, particularly in the context of asymmetric (skewed) data.
* **Data Generation**: Uses **Chi-square noise** ($\epsilon \sim \chi^2, df=2$) to introduce significant skewness into the response variable.
* **$r$-Effect Analysis**: Demonstrates how varying $r$ (e.g., comparing $r=0.1$ and $r=0.5$) shifts the "tube" to prioritize regions of higher density.
* **Visualization**: Includes comparative plots showing how different $r$ values affect the final interval width and coverage for the same skewed dataset.

## Technical Methodology

### Model Architecture
The experiments utilize a standard feed-forward MLP architecture:
* **Input Layer**: 1 unit (the $X$ feature).
* **Hidden Layer**: 200 neurons with **ReLU** activation and Random Normal kernel initialization.
* **Output Layer**: 2 neurons with **Linear** activation, representing the lower ($f_L$) and upper ($f_U$) bounds of the PI.
* **Critical Initialization**: To prevent the interval from collapsing early in training, the output biases are initialized to a wide range (e.g., `[-3, 3]`).

### Loss Function Implementation
The Tube Loss (referred to as `confidence_loss` in the code) is implemented using the `tensorflow.keras` backend. It calculates the penalty based on:
1.  **Interval Width**: The distance between $f_U$ and $f_L$.
2.  **Violation Penalty**: The degree to which the true target $y$ falls outside the estimated "tube".
3.  **The $r$ Parameter**: Controls the vertical bias of the interval.

### Optimization
* **Optimizer**: Adam optimizer with a custom learning rate schedule (Exponential Decay).
* **Training Data**: 3,000 samples for symmetric noise tests and 1,000 samples for skewed noise tests.

## How to Use
1.  **Dependencies**: Ensure `tensorflow`, `keras`, `numpy`, `matplotlib`, and `scipy` are installed.
2.  **Running Experiments**: Open the notebooks in a Jupyter environment (or Google Colab).
3.  **Configuration**: In `Tube_loss_ANN.ipynb`, you can toggle between `uniform` and `normal` noise to observe the model's robustness. In the $r$-effect notebook, modify the `r` and `delta` variables to see their influence on the interval's position and width.

## Results Visualization
The notebooks produce plots showing:
* **Scatter Plot**: The raw noisy data points.
* **True PI (Dashed Lines)**: The theoretical bounds of the noise distribution.
* **Estimated PI (Solid Lines)**: The bounds predicted by the Tube Loss-trained ANN.
