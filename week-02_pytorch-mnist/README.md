
---

## ✅ Week 2: PyTorch Basics (MNIST Classifier)

```markdown
# 🔢 Week 2 – PyTorch Basics: MNIST Digit Classifier

This week kicks off **deep learning** using `PyTorch`.  
I built a simple neural network to classify handwritten digits from the MNIST dataset.

---

## 📌 Objectives
- Learn PyTorch syntax and tensor operations
- Build and train a feedforward neural network
- Understand data loaders, optimizers, and loss functions

---

## 🧠 Concepts Covered
- Tensors, autograd, and dynamic graphs
- Model definition using `nn.Module`
- Training loops and backpropagation
- Accuracy tracking and plotting
- GPU training basics (`cuda`)

---

## 🛠️ Project

### `mnist_classifier.py`
- Loads MNIST via `torchvision`
- Builds a 3-layer dense neural network
- Trains the model with Adam optimizer
- Tracks and plots loss + accuracy

---

## 📊 Visuals
- `loss_plot.png`: Loss curve over epochs
- `accuracy_plot.png`: Accuracy progression

---

## 🧰 Tools & Libraries
- `torch`, `torchvision`, `matplotlib`

Install with:
```bash
pip install -r requirements.txt
