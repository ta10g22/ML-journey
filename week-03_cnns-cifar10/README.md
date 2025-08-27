# TinyResNet on CIFAR-10
- Model: 3-stage TinyResNet (~0.7M params), BasicBlock x2 per stage
- Training: Adam, lr=0.001, CosineAnnealingLR, 20 epochs, batch=128, aug: crop+flip
- Results: Test Acc = XX.XX%
- Artifacts: `tinyresnet_best.pt`, loss/acc curves, confusion matrix, misclassified samples


i've also included two dedicated python scripts (train.py & eval.py) where you can specify hyperparameters and run to train and evaluate your model!
 
-To train your model run this command in terminal:
python train.py --epochs 20 --bs 128 --lr 1e-3 --mixed

-To evaluate best model on test data run:
python eval.py --ckpt tinyresnet_best.pt --data data --outdir plots
