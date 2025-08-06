import torch, time
import torch.nn as nn 
import argparse

from torchvision import transforms, datasets
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()


def get_data(BATCH_SIZE):
    #function to transform data to tensors
    transform = transforms.ToTensor()
    
    #Download MNIST Dataset
    train_ds = datasets.MNIST(root ="data", train = True, download=True, transform=transform)
    test_ds =  datasets.MNIST(root = "data", train = False,download=True, transform=transform )
    print("Train size:", len(train_ds) , "test size::", len(test_ds))

    #Create batch loaders
    train_loader = DataLoader(train_ds, batch_size= BATCH_SIZE, shuffle = True)
    test_loader = DataLoader(test_ds,  batch_size = BATCH_SIZE ,shuffle = False)

    return train_loader , test_loader


def get_model():
    #Define the multi layer perceptron model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784,128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64,10)
    )
    #Choosing a loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    return loss_fn , optimizer , model


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    running_loss, running_corrects, total = 0.0, 0, 0

    for X_batch, y_batch in dataloader:                           #loop through data in barch
        X_batch, y_batch = X_batch.to(device), y_batch.to(device) #move data to right device

        optimizer.zero_grad()                                    #clear out gradient for every parameter(.grad) before a new iteration
        logits = model(X_batch)                                  # make predictions on input
        loss = loss_fn(logits, y_batch)                
        loss.backward()                                          #calculate all the gradients for parameters using back propagation
        optimizer.step()                                         #updates the parameter values with the adam optimizer

        # accumulate stats
        running_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(logits, dim=1)
        running_corrects += (preds == y_batch).sum().item()
        total += X_batch.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = running_corrects / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    val_loss, val_corrects, total = 0.0, 0, 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            val_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            val_corrects += (preds == y_batch).sum().item()
            total += X_batch.size(0)

    epoch_loss = val_loss / total
    epoch_acc  = val_corrects / total
    return epoch_loss, epoch_acc   

def plot_results(num_epochs,history):
    import matplotlib.pyplot as plt

    epochs = range (1 , num_epochs+1)

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'],   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss Curve')

    plt.subplot(1,2,2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'],   label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy Curve')

    plt.tight_layout()
    plt.show()


def main():

    device = torch.device('cpu')  # or 'cuda' if you have a GPU
    num_epochs = args.epochs
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}

    train_loader, test_loader = get_data(args.batch_size)

    loss_fn, optimizer, model = get_model()

    start = time.time()
    for epoch in range(1, num_epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss,   val_acc   = evaluate(model, test_loader,    loss_fn, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch}/{num_epochs}"
            f" — Train: loss={train_loss:.4f}, acc={train_acc:.4f}"
            f" — Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")
    end = time.time()

    print(f"Training time: {end - start: .2f} seconds")

    plot_results(num_epochs, history) 

if __name__ == '__main__' :
    main()


