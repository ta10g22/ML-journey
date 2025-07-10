# 🔢 Week 2 – PyTorch Basics: MNIST Digit Classifier

Welcome to Week 2 of my ML Journey!  
This week I dove into **PyTorch** and built a neural network to classify handwritten digits using the classic **MNIST** dataset.

---

## 📌 Objectives

- Understand PyTorch's tensor mechanics and model building blocks
- Build a fully connected neural network from scratch
- Train on MNIST with forward + backward passes
- Visualize training performance

---

## 🧠 Concepts Covered

- PyTorch Tensors, Autograd, and `nn.Module`
- Forward pass, loss computation, backpropagation
- Optimizers (`Adam`)
- Using `DataLoader` for batch processing
- GPU acceleration with `.to("cuda")`

---

## 🛠️ Project

### `mnist_classifier.py`
> Feedforward neural network (2 hidden layers) trained to classify digits (0–9)

**Steps:**
- Load MNIST with `torchvision.datasets`
- Normalize and batch with `DataLoader`
- Define a 3-layer MLP using `nn.Sequential`
- Train and evaluate the model over 5+ epochs
- Plot training loss and test accuracy

---

## 📊 Visuals

- `loss_plot.png` – Loss curve over time
- `accuracy_plot.png` – Accuracy improvement per epoch

---

## 🧰 Tools & Libraries

- Python 3.10+
- `torch`, `torchvision`
- `matplotlib`

Install with:

```bash
pip install torch torchvision matplotlib
