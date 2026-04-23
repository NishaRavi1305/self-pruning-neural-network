# Adaptive Self-Pruning Neural Network using PyTorch

## Overview

This project implements a custom self-pruning neural network that automatically identifies and suppresses unnecessary weights during training.

Unlike traditional pruning methods that remove parameters after training, this model learns which connections are useful while training itself. This is achieved through learnable gating parameters attached to each weight.

The project also studies the tradeoff between:

* Model Accuracy
* Network Sparsity (percentage of pruned weights)

using different sparsity regularization strengths.

---

## Objective

To build an efficient neural network that can reduce unnecessary connections during training while maintaining strong classification performance.

---

## Dataset Used

**CIFAR-10**

A standard image classification dataset containing 60,000 color images across 10 classes:

* Airplane

* Automobile

* Bird

* Cat

* Deer

* Dog

* Frog

* Horse

* Ship

* Truck

* Training Images: 50,000

* Test Images: 10,000

---

## Model Architecture

**Input Size:**
32 × 32 × 3 = 3072 features

**Network Structure:**

* Input Layer: 3072 neurons
* Hidden Layer: 256 neurons
* ReLU Activation
* Output Layer: 10 classes

---

## Self-Pruning Mechanism

Each weight in the custom linear layer is paired with a learnable gate score.

Gate values are computed using:

```python
gate = sigmoid(score)
```

Effective weights used during forward pass:

```python
effective_weight = weight * gate
```

If a gate value becomes very small, that connection is effectively disabled.

---

## Loss Function

Total Loss:

```python
CrossEntropyLoss + λ × SparsityPenalty
```

Where:

* **CrossEntropyLoss** improves classification accuracy
* **SparsityPenalty** encourages fewer active weights
* **λ (lambda)** controls pruning strength

---

## Experiments Performed

The network was trained using multiple lambda values:

* 0.0001
* 0.001
* 0.01

This helps analyze the balance between:

* Higher Accuracy
* Higher Compression

---

## Key Features

* Custom PyTorch pruning layer
* Learnable gates for each weight
* Dynamic pruning threshold across epochs
* Multi-lambda comparison study
* Layer-wise sparsity analysis
* Automatic device support:

  * Apple Silicon (MPS)
  * CUDA GPU
  * CPU

---

## Output Files

* `main.py` → Full implementation
* `results.png` → Training loss comparison graph
* `results.csv` → Raw experiment data
* `results_table.png` → Visual summary table

---

## How to Run

Install dependencies:

```bash
pip install torch torchvision matplotlib pandas
```

Run the project:

```bash
python3 main.py
```

---

## Results Summary

The model demonstrates the expected pruning tradeoff:

* Lower lambda values retain more weights and improve accuracy
* Higher lambda values increase sparsity but may reduce accuracy

This confirms the balance between model efficiency and predictive performance.

---


---

## Author

**Nisha Ravi**
