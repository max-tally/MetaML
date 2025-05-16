import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import math
import pandas as pd
import matplotlib.pyplot as plt

from CSVDataset import CSVDataset

def run():
    # get the gpu device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_cpu = torch.device('cpu')
    print(f"Using device: {device}")
    dataset = CSVDataset("D:/Data/Analyses/MsV1V28045_smoke.csv", "D:/Data/Analyses/Sheet8045.csv", "smoke")
    model = nn.Sequential(
        nn.Linear(len(dataset.fheader)-1, 1000),
        nn.ReLU(),
        nn.Linear(1000, 500),
        nn.ReLU(),
        nn.Linear(500, 100),
        nn.ReLU(),
        nn.Linear(100, 1)
    )
    model = model.float().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0000005,weight_decay=0.5)
    print("Starting training!")
    # Training loop
    epochs = 100
    for epoch in range(epochs):
        # Forward pass
        start_time = time.perf_counter()
        if epoch > 0:
            dataset = CSVDataset("D:/Data/Analyses/MsV1V28045_smoke.csv", "D:/Data/Analyses/Sheet8045.csv", "smoke")
        dataloader = DataLoader(dataset, batch_size=32)
        for inputs, labels in dataloader:
            y_pred = model(inputs.to(device))
            loss = criterion(y_pred.squeeze(), labels.to(device))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        end_time = time.perf_counter()
        non_zero_w= torch.sum(torch.gt(torch.abs(model.state_dict()['0.weight']),0.01).int())
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.3f} #Non-Zero: {non_zero_w}. Took: {((end_time-start_time)/60):.1f} mins.')

    torch.save(model,"SmokingModel.pth")

if __name__ == '__main__':
    run()
