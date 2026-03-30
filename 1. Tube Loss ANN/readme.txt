Implementation of the Tube loss based NN

A simple artificial dataset illustration ( Figure 8 of the paper) is at Tube_loss_ANN.ipynb
The effect of the r parameter at Tube loss is at Tube_loss_ANN_r_effect.ipynb  


# Tube Loss ANN - Experiments

For a detailed overview of the Tube Loss function, please refer to the [main README](../README.md) and the full paper at https://arxiv.org/pdf/2412.06853

## Experiments in This Folder

### Experiment 1: Basic Tube Loss ANN Demonstration (`Tube_loss_ANN.ipynb`)

**Purpose**: Illustrates Figure 8 from the paper - demonstrating how Tube Loss trains an ANN to produce well-calibrated prediction intervals on a simple artificial dataset.

#### Dataset
- **Type**: Artificial synthetic data
- **Function**: $y = \frac{\sin(x)}{x} + \text{noise}$
- **Samples**: 3000 points
- **Noise**: Uniform distribution [-1, 1] (configurable to normal distribution)

#### Model Architecture
```python
Input Layer (1 feature) 
    ↓
Hidden Layer: 100 neurons, ReLU activation
    ↓
Output Layer: 2 neurons, Linear activation (upper bound, lower bound)
```

**Key Design Decision**: Output layer biases initialized to `[-3, 3]` to ensure prediction intervals start with reasonable bounds.

#### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `q` | 0.95 | Target coverage probability (95%) |
| `r` | 0.5 | Tube movement parameter (optimal for symmetric noise) |
| `δ` (delta) | 0 | Recalibration term |
| Batch size | 100 | Mini-batch size |
| Initial learning rate | 0.02 | Adam optimizer with exponential decay |

#### Training Configuration
- **Optimizer**: Adam with exponential learning rate decay
- **Decay schedule**: decay_steps=10000, decay_rate=0.01
- **Total epochs**: 1200 (shown in 6 iterations of 200 epochs each)
- **Loss function selection**: `l=2` for Tube Loss, `l=1` for QD loss

#### Evaluation Metrics
```python
PICP = mean(y_true ∈ [y_lower_pred, y_upper_pred])  # Coverage probability
MPIW = mean(y_upper_pred - y_lower_pred)            # Mean interval width
```

---

### Experiment 2: Effect of r Parameter (`Tube_loss_ANN_r_effect.ipynb`)

**Purpose**: Demonstrates how the `r` parameter enables the PI tube to move up and down, capturing denser regions of data clouds for narrower prediction intervals.

#### Key Concept
The `r` parameter controls the movement direction of the prediction interval tube:
- **r = 0.5**: Optimal for symmetric noise distributions (centered tube)
- **r < 0.5**: Tube moves downward - beneficial for long-tailed distributions with upper outliers
- **r > 0.5**: Tube moves upward - beneficial for long-tailed distributions with lower outliers

#### Practical Guidance
> For symmetric noise, use r=0.5. For asymmetric/long-tailed noise, tune r to achieve narrower PIs while maintaining target coverage.

---

### Comparison: Tube Loss vs QD Loss

The notebook includes implementation of the **QD (Quantile-based Deep) loss** as a baseline:

| Aspect | Tube Loss | QD Loss |
|--------|-----------|---------|
| Continuity | Fully continuous | Uses sigmoid approximation |
| Coverage guarantee | Asymptotic proof provided | Heuristic approach |
| Parameters | q, r, δ | λ, α, soften |
| Training stability | Direct gradient flow | Requires tuning of λ |

**QD Loss Hyperparameters in Code**:
- `lambda_ = 0.01` (penalty weight)
- `alpha_ = 0.05` (targeting 95% coverage)
- `soften_ = 255.0` (sigmoid steepness for approximation)

---

### Visual Output Description

**Figure from Tube_loss_ANN.ipynb**:
- **Red scatter points**: Training data with noise
- **Black dashed lines**: True prediction interval boundaries
- **Blue solid lines**: Estimated prediction interval by trained model

The visualization demonstrates how the learned PI tube converges toward the true PI after 1200 epochs of training.

---

### How to Run

```python
# In Jupyter notebook or Google Colab:
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

# Load and execute cells from Tube_loss_ANN.ipynb
```

**Requirements**:
- Python 3.7+
- TensorFlow/Keras ≥ 2.0
- NumPy
- Matplotlib

---

### Expected Results

For the uniform noise dataset with q=0.95:
- **PICP (Coverage)**: ≈ 0.94 - 0.96 (target: 0.95)
- **MPIW (Width)**: ≈ 1.8 - 2.2 (depends on training convergence)

Tube Loss typically achieves tighter intervals than QD loss while maintaining comparable or better coverage probability.

---

<!-- 
================================================================================
ORIGINAL README.TXT CONTENTS (PRESERVED FOR REFERENCE)
================================================================================

Implementation of the Tube loss based NN

A simple artificial dataset illustration ( Figure 8 of the paper) is at Tube_loss_ANN.ipynb
The effect of the r parameter at Tube loss is at Tube_loss_ANN_r_effect.ipynb  

-->
