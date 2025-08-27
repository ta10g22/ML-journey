import argparse, time, os, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ----- TinyResNet -----
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
            self.downsample = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                                            nn.BatchNorm2d(out_ch))
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None: identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)

class TinyResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3,32,3,1,1,bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
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
        for _ in range(1, blocks): layers.append(BasicBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.pool(x); x = torch.flatten(x, 1); return self.fc(x)

# ----- Train/Eval helpers -----
@torch.no_grad()
def evaluate(model, loader, crit, device):
    model.eval(); loss_sum=0.0; correct=0; total=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x); loss = crit(logits, y)
        loss_sum += loss.item()*x.size(0); correct += (logits.argmax(1)==y).sum().item(); total += x.size(0)
    return loss_sum/total, correct/total

def train_one_epoch(model, loader, opt, crit, device, scaler=None):
    model.train(); loss_sum=0.0; correct=0; total=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(x); loss = crit(logits, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            logits = model(x); loss = crit(logits, y); loss.backward(); opt.step()
        loss_sum += loss.item()*x.size(0); correct += (logits.argmax(1)==y).sum().item(); total += x.size(0)
    return loss_sum/total, correct/total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--data', type=str, default='data')
    ap.add_argument('--out', type=str, default='tinyresnet_best.pt')
    ap.add_argument('--mixed', action='store_true', help='use torch.cuda.amp if CUDA')
    args = ap.parse_args()

    torch.manual_seed(42); np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available()
                          else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465); CIFAR10_STD = (0.2470, 0.2435, 0.2616)
    train_tfms = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    test_tfms  = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])

    train_set = datasets.CIFAR10(args.data, train=True,  download=True, transform=train_tfms)
    test_set  = datasets.CIFAR10(args.data, train=False, download=True, transform=test_tfms)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True,  num_workers=2, pin_memory=pin)
    test_loader  = DataLoader(test_set,  batch_size=args.bs, shuffle=False, num_workers=2, pin_memory=pin)

    model = TinyResNet(num_classes=10).to(device)
    crit = nn.CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if (args.mixed and torch.cuda.is_available()) else None

    best_acc=0.0; t0=time.time()
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, crit, device, scaler)
        va_loss, va_acc = evaluate(model, test_loader, crit, device)  # use a val split in real projects
        sched.step()
        if va_acc > best_acc:
            best_acc = va_acc; torch.save(model.state_dict(), args.out)
        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"Train loss {tr_loss:.4f} acc {tr_acc*100:5.2f}% | "
              f"Val loss {va_loss:.4f} acc {va_acc*100:5.2f}%")

    print(f"Best Val Acc: {best_acc*100:.2f}% | Time: {time.time()-t0:.1f}s | Saved -> {args.out}")

if __name__ == "__main__":
    main()
