# TinyResNet on CIFAR-10

- **Model**: 3-stage TinyResNet (~0.7M params), BasicBlock ×2 per stage  
- **Training**: Adam, lr=0.001, CosineAnnealingLR, 20 epochs, batch=128, aug: crop+flip  
- **Results**: Test Acc = XX.XX%  
- **Artifacts**: `tinyresnet_best.pt`, loss/acc curves, confusion matrix, misclassified samples  

Two scripts are provided: `train.py` (train the model) and `eval.py` (evaluate a checkpoint).

---

## Setup

```bash
# clone repo and enter folder
git clone <your-repo-url>
cd <your-repo-folder>

# create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate    # (Windows PowerShell: .venv\Scripts\Activate.ps1, Mac/Linux: source .venv/bin/activate)

# install dependencies
pip install -r requirements.txt

# training the model
python train.py --epochs 20 --bs 128 --lr 1e-3 --mixed

# evaluating the model 
python eval.py --ckpt tinyresnet_best.pt --data data --outdir plots

"""
This saves:
confusion matrix → plots/confusion_matrix.png
worst-6 misclassified → plots/worst6.png
per-class accuracy printed in terminal
"""