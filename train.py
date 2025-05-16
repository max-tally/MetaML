import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import pandas as pd
import matplotlib.pyplot as plt

#import CSVDataset

def get_model_memory_size(model):
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    total_mem = mem_params + mem_bufs
    return total_mem / (1024**2)


def run():
    # get the gpu device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_cpu = torch.device('cpu')
    print(f"Using device: {device}")

    # we will load in the data
    ms = pd.read_csv("D:/Data/Analyses/Ms_250kx2k.csv",low_memory=False)
    ms = ms.astype(float)
    ms = ms.dropna(axis=1)

    ms2 = pd.read_csv("D:/Data/Analyses/Ms_250kx2k_2.csv",low_memory=False)
    ms2 = ms2.astype(float)
    ms2 = ms2.dropna(axis=1)


    meta = pd.read_csv("D:/Data/Analyses/Meta_2k.csv")
    meta2 = pd.read_csv("D:/Data/Analyses/Meta_2k_2.csv")
    print("Loaded the data and metadata!")

    X = torch.tensor(ms.values).float()
    y = torch.tensor(meta['chronological_age'].values).float()

    X2 = torch.tensor(ms2.values).float()
    X2=X2.to(device)
    y2 = torch.tensor(meta2['chronological_age'].values).float()
    y2=y2.to(device)
    # let's create a simple sequential PyTorch model
    # Define the model
    model = nn.Sequential(
        nn.Linear(X.shape[1], 500),
        nn.ReLU(),
        nn.Linear(500, 100),
        nn.ReLU(),
        nn.Linear(100, 1)
    )
    model=model.float().to(device)
    print(f"Model size {get_model_memory_size(model):.2f} Mb")

    # Create a dataset and dataloader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.000007,weight_decay=0.1)
    print("Starting training!")
    # Training loop
    epochs = 250
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs.to(device))

            # Calculate the loss
            loss = criterion(outputs.squeeze(), labels.to(device))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        voutputs = model(X2)
        vloss = criterion(voutputs.squeeze(), y2)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {math.sqrt(loss.item()):.3f}, vLoss: {math.sqrt(vloss.item()):.3f}')
    model.to(device_cpu)
    voutputs = model(X2.to(device_cpu))
    xs= y2.to(device_cpu)[1:100].detach().numpy()
    ys= voutputs[1:100].detach().numpy()
    plt.scatter(xs, ys, s=50, c='blue', marker='o', label='DL Test')
    # Add labels and title
    plt.xlabel('Chron Age')
    plt.ylabel('DL Age')
    plt.title('Test Dataset')
    # Show the plot
    plt.show()

if __name__ == '__main__':
    run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
