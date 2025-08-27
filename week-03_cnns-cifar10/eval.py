# eval.py
import os, argparse, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ---------- TinyResNet (same as train) ----------
def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride); self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = conv3x3(out_ch, out_ch, 1);     self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)

class TinyResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3,32,3,1,1,bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(32, 32, blocks=2, stride=1)
        self.layer2 = self._make_layer(32, 64, blocks=2, stride=2)
        self.layer3 = self._make_layer(64,128, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.pool(x); x = torch.flatten(x, 1); return self.fc(x)

# ---------- Utils ----------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def get_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    if torch.backends.mps.is_available() and torch.backends.mps.is_built(): return torch.device('mps')
    return torch.device('cpu')

def build_test_loader(data_dir, bs=128):
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=tfm)
    pin = torch.cuda.is_available()
    loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=2, pin_memory=pin)
    return loader, test_set.classes

@torch.no_grad()
def collect_logits_preds_targets(model, loader, device):
    model.eval()
    all_logits, all_preds, all_tgts = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_preds.append(logits.argmax(1).cpu())
        all_tgts.append(y.cpu())
    return (torch.cat(all_logits), torch.cat(all_preds), torch.cat(all_tgts))

def save_confusion_matrix(tgts, preds, class_names, outpath, normalize=False):
    cm = confusion_matrix(tgts, preds)
    if normalize:
        with np.errstate(all='ignore'):
            cm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    plt.figure(figsize=(6.5,6))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix" + (" (norm)" if normalize else ""))
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha='right')
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def unnormalize(img_t):
    mean = torch.tensor(CIFAR10_MEAN).view(3,1,1)
    std  = torch.tensor(CIFAR10_STD).view(3,1,1)
    return (img_t*std + mean).clamp(0,1)

def save_worst6_grid(model, loader, class_names, device, outpath):
    worst = []  # list of (conf, img_cpu, true, pred)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs  = F.softmax(logits, dim=1)
            conf, preds = probs.max(dim=1)
            mis = preds.cpu() != y
            if mis.any():
                mis_idx = mis.nonzero(as_tuple=False).squeeze(1)
                for i in mis_idx:
                    worst.append((
                        conf[i].item(),
                        x[i].detach().cpu(),
                        int(y[i].item()),
                        int(preds[i].cpu().item())
                    ))
    if not worst:
        print("No misclassifications found — great job!"); return
    worst.sort(key=lambda t: t[0], reverse=True)
    top6 = worst[:6]

    cols, rows = 3, 2
    plt.figure(figsize=(12, 6))
    for k, (c, img, t, p) in enumerate(top6):
        plt.subplot(rows, cols, k+1)
        img_disp = unnormalize(img).permute(1,2,0).numpy()
        plt.imshow(img_disp)
        plt.title(f"True: {class_names[t]}\nPred: {class_names[p]} ({c*100:.1f}%)")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def per_class_accuracy(tgts, preds, class_names):
    tgts = np.asarray(tgts); preds = np.asarray(preds)
    totals = np.bincount(tgts, minlength=len(class_names))
    corrects = np.bincount(tgts[preds==tgts], minlength=len(class_names))
    for i, name in enumerate(class_names):
        if totals[i] == 0: acc = 0.0
        else: acc = 100.0 * corrects[i] / totals[i]
        print(f"{name:12s}: {acc:5.1f}%  ({corrects[i]}/{totals[i]})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='path to model .pt (e.g., tinyresnet_best.pt)')
    ap.add_argument('--data', default='data', help='CIFAR-10 data dir')
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--outdir', default='plots')
    ap.add_argument('--norm_cm', action='store_true', help='normalize confusion matrix')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    # Data
    test_loader, classes = build_test_loader(args.data, bs=args.bs)

    # Model
    model = TinyResNet(num_classes=10).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Eval
    logits, preds, tgts = collect_logits_preds_targets(model, test_loader, device)
    test_acc = (preds == tgts).float().mean().item() * 100.0
    print(f"Test accuracy: {test_acc:.2f}%")

    # Confusion matrix
    cm_path = os.path.join(args.outdir, 'confusion_matrix.png' if not args.norm_cm else 'confusion_matrix_norm.png')
    save_confusion_matrix(tgts.numpy(), preds.numpy(), classes, cm_path, normalize=args.norm_cm)
    print(f"Saved confusion matrix -> {cm_path}")

    # Worst 6 mispredictions
    worst6_path = os.path.join(args.outdir, 'worst6.png')
    save_worst6_grid(model, test_loader, classes, device, worst6_path)
    print(f"Saved worst-6 grid -> {worst6_path}")

    # Per-class accuracy (nice for README)
    print("\nPer-class accuracy:")
    per_class_accuracy(tgts.numpy(), preds.numpy(), classes)

if __name__ == '__main__':
    main()
