import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from CSVDataset import CSVDataset


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, weights):
        return (weights * (input - target) ** 2).mean()

class SqrtLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sqrt(x)


class SmokingModel(nn.Module):
    def __init__(self, num_inputs):
        super(SmokingModel, self).__init__()
        # Define the common input layer
        self.input_drop = nn.Dropout(0.2,inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.sqrt1 = SqrtLayer()

        self.decoder = nn.Sequential(
            nn.Linear(num_inputs*2,1000),
            nn.ReLU(),
            nn.Linear(1000, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        # Pass input through the common input layer
        x = self.input_drop(x)
        x = self.sigmoid(x)

        output1 = self.sqrt1(x)

        # Pass the input through both branches
        # Concatenate the outputs of the branches
        combined_output = torch.cat((output1, x), dim=1)

        # Pass the combined output through the output layer
        final_output = self.decoder(combined_output)
        return final_output

def runSmoking():
    # get the gpu device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    ms = pd.read_csv("H:/My Drive/Data/classifier/ms8k_smoke.csv", encoding='cp1252',index_col=0)
    meta = pd.read_csv("H:/My Drive/Data/classifier/meta.csv", encoding='cp1252',index_col=0)

    cors = ms.apply(lambda col: np.corrcoef(col, meta['smoke'])[0, 1])
    ms = ms.iloc[:,np.argsort(abs(cors))[::-1]].iloc[:, 0:10000]

    #meta = meta2.iloc[even, :]
    print(f'Ms: {ms.shape}; meta: {meta.shape}')
    # Initialize the k-fold cross validation
    # kf = KFold(n_splits=k_folds, shuffle=True)

    # dataset=TensorDataset(torch.tensor(ms.values).float(),torch.tensor(meta['smoke'].values).float())



    # Instantiate the model
    model = SmokingModel(ms.shape[1])
    model = model.float().to(device)
    optimizer = optim.Adam(model.parameters(),lr=1e-7)
    # class_weights = torch.tensor([0.9,0.1])
    criterion = WeightedMSELoss()
    mse = nn.MSELoss()
    print("Starting training!")
    num_samples = 7000
    train = torch.randperm(ms.shape[0])[:num_samples].numpy()  # Shuffle indices and take the first num_samples

    test = [x for x in range(8045) if x not in train]

    X = torch.tensor(ms.iloc[train, :].values).float()
    y = torch.tensor(meta['smoke'].iloc[train].values).float()
    X2 = torch.tensor(ms.iloc[test, :].values).float()
    y2 = torch.tensor(meta['smoke'].iloc[test].values).float()

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=128)

    epochs = 2500
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs.to(device))
            # weights
            weights = torch.divide(1.0,torch.add(labels,0.25))
            # weights = torch.ones(len(labels))

            # Calculate the loss
            loss = criterion(outputs.squeeze(), labels.to(device),weights.to(device))
            #loss = criterion(outputs.squeeze(), labels.to(device))
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        voutputs = model(X2.to(device))
        # trainout = model(X.to(device))

        vloss = mse(voutputs.squeeze(), y2.to(device))

        print(
            f'Epoch [{epoch + 1}/{epochs}], wMSE: {loss.item():.6f}, vMSE: {vloss.item():.6f}')
        if epoch % 500 == 0:
            y_pred=voutputs.squeeze().cpu().detach().numpy()
            y_test=y2.cpu().detach().numpy()
            z = np.polyfit(y_test, y_pred, 1)
            p = np.poly1d(z)
            plt.plot(y_test,y_pred,'o', label='Test data')
            plt.plot(y_test,p(y_test),label=f'Trend: y={z[0]:.3f}x + {z[1]:.3f}')
            plt.title(f"Epoch {epoch+1}")
            plt.legend()
            plt.pause(3)
            plt.show(block=False)
            if epoch+500 < epochs:
                plt.close()

    torch.save(model.state_dict(), "D:/Data/Analyses/SmokingModel.pth")


def runAlcohol():
    # get the gpu device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    ms = pd.read_csv("H:/My Drive/Data/classifier/ms8k_drink.csv", encoding='cp1252',index_col=0)
    meta = pd.read_csv("H:/My Drive/Data/classifier/meta.csv", encoding='cp1252',index_col=0)

    print(f'Ms: {ms.shape}; meta: {meta.shape}')
    # Initialize the k-fold cross validation
    model = nn.Sequential(
        nn.Linear(ms.shape[1], 2000),
        nn.ReLU(),
        nn.Linear(2000, 500),
        nn.ReLU(),
        nn.Linear(500, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    model = model.float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    # class_weights = torch.tensor([0.9,0.1])
    # criterion = WeightedMSELoss()
    criterion = nn.MSELoss()
    print("Starting training!")
    num_samples = 7000
    train = torch.randperm(ms.shape[0])[:num_samples].numpy()  # Shuffle indices and take the first num_samples

    test = [x for x in range(8045) if x not in train]

    X = torch.tensor(ms.iloc[train, :].values).float()
    y = torch.tensor(meta['drink'].iloc[train].values).float()
    X2 = torch.tensor(ms.iloc[test, :].values).float()
    y2 = torch.tensor(meta['drink'].iloc[test].values).float()

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=128)

    epochs = 10000
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs.to(device))
            # weights
            # weights = torch.divide(1.0,torch.add(labels,0.1))
            # weights = torch.ones(len(labels))

            # Calculate the loss
            # loss = criterion(outputs.squeeze(), labels.to(device),weights.to(device))
            loss = criterion(outputs.squeeze(), labels.to(device))
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        voutputs = model(X2.to(device))
        # trainout = model(X.to(device))

        vloss2 = criterion(voutputs.squeeze(), y2.to(device))
        if epoch % 10 == 0:
            print(
                f'Epoch [{epoch + 1}/{epochs}], MSE: {loss.item():.6f}, vMSE: {vloss2.item():.6f}, Mean drinkers:{torch.mean(voutputs.squeeze()[y2 < 0.25]):.6f} Mean abstainers:{torch.mean(voutputs.squeeze()[y2 > 0.75]):.6f}')
        if epoch % 500 == 0:
            y_pred = voutputs.squeeze().cpu().detach().numpy()
            y_test = y2.cpu().detach().numpy()
            z = np.polyfit(y_test, y_pred, 1)
            p = np.poly1d(z)
            plt.plot(y_test, y_pred, 'o', label='Test data')
            plt.plot(y_test, p(y_test), label=f'Trend: y={z[0]:.3f}x + {z[1]:.3f}')
            plt.title(f"Alcohol Epoch {epoch + 1}")
            plt.legend()
            plt.pause(3)
            plt.show(block=False)
            plt.close()
    y_pred = voutputs.squeeze().cpu().detach().numpy()
    y_test = y2.cpu().detach().numpy()
    z = np.polyfit(y_test, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_test, y_pred, 'o', label='Test data')
    plt.plot(y_test, p(y_test), label=f'Trend: y={z[0]:.3f}x + {z[1]:.3f}')
    plt.title("Alcohol Deep Learning Model")
    plt.legend()
    plt.savefig('H:/My Drive/Data/classifier/Alcohol_deep.png')
    plt.pause(3)
    plt.show(block=False)
    torch.save(model.state_dict(), "H:/My Drive/Data/classifier/AlcoholModel.pth")

def runBMI():
    # get the gpu device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    ms = pd.read_csv("H:/My Drive/Data/classifier/ms8k_bmi.csv", encoding='cp1252',index_col=0)
    meta = pd.read_csv("H:/My Drive/Data/classifier/meta.csv", encoding='cp1252',index_col=0)

    print(f'Ms: {ms.shape}; meta: {meta.shape}')
    # Initialize the k-fold cross validation
    model = nn.Sequential(
        nn.Linear(ms.shape[1], 2000),
        nn.ReLU(),
        nn.Linear(2000, 500),
        nn.ReLU(),
        nn.Linear(500, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    model = model.float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    # class_weights = torch.tensor([0.9,0.1])
    # criterion = WeightedMSELoss()
    criterion = nn.MSELoss()
    print("Starting training!")
    num_samples = 7000
    train = torch.randperm(ms.shape[0])[:num_samples].numpy()  # Shuffle indices and take the first num_samples

    test = [x for x in range(8045) if x not in train]

    X = torch.tensor(ms.iloc[train, :].values).float()
    y = torch.tensor(meta['BMI'].iloc[train].values).float()
    X2 = torch.tensor(ms.iloc[test, :].values).float()
    y2 = torch.tensor(meta['BMI'].iloc[test].values).float()

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=128)

    epochs = 10000
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs.to(device))
            # weights
            # weights = torch.divide(1.0,torch.add(labels,0.1))
            # weights = torch.ones(len(labels))

            # Calculate the loss
            # loss = criterion(outputs.squeeze(), labels.to(device),weights.to(device))
            loss = criterion(outputs.squeeze(), labels.to(device))
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        voutputs = model(X2.to(device))
        # trainout = model(X.to(device))

        vloss2 = criterion(voutputs.squeeze(), y2.to(device))
        if epoch % 10 == 0:
            print(
                f'Epoch [{epoch + 1}/{epochs}], MSE: {loss.item():.6f}, vMSE: {vloss2.item():.6f}, Mean BMI < 20:{torch.mean(voutputs.squeeze()[y2 < 20]):.6f} Mean BMI>30:{torch.mean(voutputs.squeeze()[y2 > 30]):.6f}')
        if epoch % 500 == 0:
            y_pred = voutputs.squeeze().cpu().detach().numpy()
            y_test = y2.cpu().detach().numpy()
            z = np.polyfit(y_test, y_pred, 1)
            p = np.poly1d(z)
            plt.plot(y_test, y_pred, 'o', label='Test data')
            plt.plot(y_test, p(y_test), label=f'Trend: y={z[0]:.3f}x + {z[1]:.3f}')
            plt.title(f"BMI Epoch {epoch + 1}")
            plt.legend()
            plt.pause(3)
            plt.show(block=False)
            plt.close()
    y_pred = voutputs.squeeze().cpu().detach().numpy()
    y_test = y2.cpu().detach().numpy()
    z = np.polyfit(y_test, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_test, y_pred, 'o', label='Test data')
    plt.plot(y_test, p(y_test), label=f'Trend: y={z[0]:.3f}x + {z[1]:.3f}')
    plt.title("BMI Deep Learning Model")
    plt.legend()
    plt.pause(3)
    plt.savefig('H:/My Drive/Data/classifier/BMI_deep.png')
    plt.show(block=False)
    torch.save(model.state_dict(), "H:/My Drive/Data/classifier/BMIModel.pth")

def runAge():
    # get the gpu device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    ms = pd.read_csv("H:/My Drive/Data/classifier/ms8k_age.csv", encoding='cp1252',index_col=0)
    meta = pd.read_csv("H:/My Drive/Data/classifier/meta.csv", encoding='cp1252',index_col=0)

    print(f'Ms: {ms.shape}; meta: {meta.shape}')
    # Initialize the k-fold cross validation
    model = nn.Sequential(
        nn.Linear(ms.shape[1], 2000),
        nn.ReLU(),
        nn.Linear(2000, 500),
        nn.ReLU(),
        nn.Linear(500, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    model = model.float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    # class_weights = torch.tensor([0.9,0.1])
    # criterion = WeightedMSELoss()
    criterion = nn.MSELoss()
    print("Starting training!")
    num_samples = 7000
    train = torch.randperm(ms.shape[0])[:num_samples].numpy()  # Shuffle indices and take the first num_samples

    test = [x for x in range(8045) if x not in train]

    X = torch.tensor(ms.iloc[train, :].values).float()
    y = torch.tensor(meta['age'].iloc[train].values).float()
    X2 = torch.tensor(ms.iloc[test, :].values).float()
    y2 = torch.tensor(meta['age'].iloc[test].values).float()

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=128)

    epochs = 10000
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs.to(device))
            # weights
            # weights = torch.divide(1.0,torch.add(labels,0.1))
            # weights = torch.ones(len(labels))

            # Calculate the loss
            # loss = criterion(outputs.squeeze(), labels.to(device),weights.to(device))
            loss = criterion(outputs.squeeze(), labels.to(device))
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        voutputs = model(X2.to(device))
        # trainout = model(X.to(device))

        vloss2 = criterion(voutputs.squeeze(), y2.to(device))
        if epoch % 10 == 0:
            print(
                f'Epoch [{epoch + 1}/{epochs}], MSE: {loss.item():.6f}, vMSE: {vloss2.item():.6f}, Mean Age < 30:{torch.mean(voutputs.squeeze()[y2 < 30]):.6f} Mean Age>70:{torch.mean(voutputs.squeeze()[y2 > 70]):.6f}')
        if epoch % 500 == 0:
            y_pred = voutputs.squeeze().cpu().detach().numpy()
            y_test = y2.cpu().detach().numpy()
            z = np.polyfit(y_test, y_pred, 1)
            p = np.poly1d(z)
            plt.plot(y_test, y_pred, 'o', label='Test data')
            plt.plot(y_test, p(y_test), label=f'Trend: y={z[0]:.3f}x + {z[1]:.3f}')
            plt.title(f"Age Epoch {epoch + 1}")
            plt.legend()
            plt.pause(3)
            plt.show(block=False)
            plt.close()
    y_pred = voutputs.squeeze().cpu().detach().numpy()
    y_test = y2.cpu().detach().numpy()
    z = np.polyfit(y_test, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_test, y_pred, 'o', label='Test data')
    plt.plot(y_test, p(y_test), label=f'Trend: y={z[0]:.3f}x + {z[1]:.3f}')
    plt.title("Age Deep Learning Model")
    plt.legend()
    plt.pause(3)
    plt.savefig('H:/My Drive/Data/classifier/Age_deep.png')
    plt.show(block=False)
    torch.save(model.state_dict(), "H:/My Drive/Data/classifier/AgeModel.pth")

def runAge2():
    # get the gpu device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    ms = pd.read_csv("H:/My Drive/Data/classifier/ms8k_age.csv", encoding='cp1252',index_col=0)
    meta = pd.read_csv("H:/My Drive/Data/classifier/meta.csv", encoding='cp1252',index_col=0)

    print(f'Ms: {ms.shape}; meta: {meta.shape}')
    # Initialize the k-fold cross validation
    model = nn.Sequential(
        nn.Linear(ms.shape[1], 1)
      )
    model = model.float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    # class_weights = torch.tensor([0.9,0.1])
    # criterion = WeightedMSELoss()
    criterion = nn.MSELoss()
    print("Starting training!")
    num_samples = 7000
    train = torch.randperm(ms.shape[0])[:num_samples].numpy()  # Shuffle indices and take the first num_samples

    test = [x for x in range(8045) if x not in train]

    X = torch.tensor(ms.iloc[train, :].values).float()
    y = torch.tensor(meta['age'].iloc[train].values).float()
    X2 = torch.tensor(ms.iloc[test, :].values).float()
    y2 = torch.tensor(meta['age'].iloc[test].values).float()

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=128)

    epochs = 10000
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs.to(device))
            # weights
            # weights = torch.divide(1.0,torch.add(labels,0.1))
            # weights = torch.ones(len(labels))

            # Calculate the loss
            # loss = criterion(outputs.squeeze(), labels.to(device),weights.to(device))
            loss = criterion(outputs.squeeze(), labels.to(device))
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        voutputs = model(X2.to(device))
        # trainout = model(X.to(device))

        vloss2 = criterion(voutputs.squeeze(), y2.to(device))
        if epoch % 10 == 0:
            print(
                f'Epoch [{epoch + 1}/{epochs}], MSE: {loss.item():.6f}, vMSE: {vloss2.item():.6f}, Mean Age < 30:{torch.mean(voutputs.squeeze()[y2 < 30]):.6f} Mean Age>70:{torch.mean(voutputs.squeeze()[y2 > 70]):.6f}')
        if epoch % 500 == 0:
            y_pred = voutputs.squeeze().cpu().detach().numpy()
            y_test = y2.cpu().detach().numpy()
            z = np.polyfit(y_test, y_pred, 1)
            p = np.poly1d(z)
            plt.plot(y_test, y_pred, 'o', label='Test data')
            plt.plot(y_test, p(y_test), label=f'Trend: y={z[0]:.3f}x + {z[1]:.3f}')
            plt.title(f"Age Epoch {epoch + 1}")
            plt.legend()
            plt.pause(3)
            plt.show(block=False)
            plt.close()
    y_pred = voutputs.squeeze().cpu().detach().numpy()
    y_test = y2.cpu().detach().numpy()
    z = np.polyfit(y_test, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_test, y_pred, 'o', label='Test data')
    plt.plot(y_test, p(y_test), label=f'Trend: y={z[0]:.3f}x + {z[1]:.3f}')
    plt.title("Age Deep Learning Model")
    plt.legend()
    plt.pause(3)
    plt.savefig('H:/My Drive/Data/classifier/Age_deep2.png')
    plt.show(block=False)
    torch.save(model.state_dict(), "H:/My Drive/Data/classifier/AgeModel2.pth")

def runAge3():
    # get the gpu device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    ms = pd.read_csv("H:/My Drive/Data/classifier/ms8k_age.csv", encoding='cp1252',index_col=0)
    meta = pd.read_csv("H:/My Drive/Data/classifier/meta.csv", encoding='cp1252',index_col=0)

    print(f'Ms: {ms.shape}; meta: {meta.shape}')
    # Initialize the k-fold cross validation
    model = nn.Sequential(
        nn.Linear(ms.shape[1], 2000),
        nn.ReLU(),
        nn.Linear(2000, 1)
    )
    model = model.float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    # class_weights = torch.tensor([0.9,0.1])
    # criterion = WeightedMSELoss()
    criterion = nn.MSELoss()
    print("Starting training!")
    num_samples = 7000
    train = torch.randperm(ms.shape[0])[:num_samples].numpy()  # Shuffle indices and take the first num_samples

    test = [x for x in range(8045) if x not in train]

    X = torch.tensor(ms.iloc[train, :].values).float()
    y = torch.tensor(meta['age'].iloc[train].values).float()
    X2 = torch.tensor(ms.iloc[test, :].values).float()
    y2 = torch.tensor(meta['age'].iloc[test].values).float()

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=128)

    epochs = 10000
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs.to(device))
            # weights
            # weights = torch.divide(1.0,torch.add(labels,0.1))
            # weights = torch.ones(len(labels))

            # Calculate the loss
            # loss = criterion(outputs.squeeze(), labels.to(device),weights.to(device))
            loss = criterion(outputs.squeeze(), labels.to(device))
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        voutputs = model(X2.to(device))
        # trainout = model(X.to(device))

        vloss2 = criterion(voutputs.squeeze(), y2.to(device))
        if epoch % 10 == 0:
            print(
                f'Epoch [{epoch + 1}/{epochs}], MSE: {loss.item():.6f}, vMSE: {vloss2.item():.6f}, Mean Age < 30:{torch.mean(voutputs.squeeze()[y2 < 30]):.6f} Mean Age>70:{torch.mean(voutputs.squeeze()[y2 > 70]):.6f}')
        if epoch % 500 == 0:
            y_pred = voutputs.squeeze().cpu().detach().numpy()
            y_test = y2.cpu().detach().numpy()
            z = np.polyfit(y_test, y_pred, 1)
            p = np.poly1d(z)
            plt.plot(y_test, y_pred, 'o', label='Test data')
            plt.plot(y_test, p(y_test), label=f'Trend: y={z[0]:.3f}x + {z[1]:.3f}')
            plt.title(f"Age Epoch {epoch + 1}")
            plt.legend()
            plt.pause(3)
            plt.show(block=False)
            plt.close()
    y_pred = voutputs.squeeze().cpu().detach().numpy()
    y_test = y2.cpu().detach().numpy()
    z = np.polyfit(y_test, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_test, y_pred, 'o', label='Test data')
    plt.plot(y_test, p(y_test), label=f'Trend: y={z[0]:.3f}x + {z[1]:.3f}')
    plt.title("Age Deep Learning Model")
    plt.legend()
    plt.pause(3)
    plt.savefig('H:/My Drive/Data/classifier/Age_deep3.png')
    plt.show(block=False)
    torch.save(model.state_dict(), "H:/My Drive/Data/classifier/AgeModel3.pth")

def runAge4():
    # get the gpu device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    ms = pd.read_csv("H:/My Drive/Data/classifier/ms8k_age.csv", encoding='cp1252',index_col=0)
    meta = pd.read_csv("H:/My Drive/Data/classifier/meta.csv", encoding='cp1252',index_col=0)

    print(f'Ms: {ms.shape}; meta: {meta.shape}')
    # Initialize the k-fold cross validation
    model = nn.Sequential(
        nn.Linear(ms.shape[1], 2000),
        nn.ReLU(),
        nn.Linear(2000, 500),
        nn.ReLU(),
        nn.Linear(500, 1)
    )
    model = model.float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    # class_weights = torch.tensor([0.9,0.1])
    # criterion = WeightedMSELoss()
    criterion = nn.MSELoss()
    print("Starting training!")
    num_samples = 7000
    train = torch.randperm(ms.shape[0])[:num_samples].numpy()  # Shuffle indices and take the first num_samples

    test = [x for x in range(8045) if x not in train]

    X = torch.tensor(ms.iloc[train, :].values).float()
    y = torch.tensor(meta['age'].iloc[train].values).float()
    X2 = torch.tensor(ms.iloc[test, :].values).float()
    y2 = torch.tensor(meta['age'].iloc[test].values).float()

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=128)

    epochs = 10000
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs.to(device))
            # weights
            # weights = torch.divide(1.0,torch.add(labels,0.1))
            # weights = torch.ones(len(labels))

            # Calculate the loss
            # loss = criterion(outputs.squeeze(), labels.to(device),weights.to(device))
            loss = criterion(outputs.squeeze(), labels.to(device))
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        voutputs = model(X2.to(device))
        # trainout = model(X.to(device))

        vloss2 = criterion(voutputs.squeeze(), y2.to(device))
        if epoch % 10 == 0:
            print(
                f'Epoch [{epoch + 1}/{epochs}], MSE: {loss.item():.6f}, vMSE: {vloss2.item():.6f}, Mean Age < 30:{torch.mean(voutputs.squeeze()[y2 < 30]):.6f} Mean Age>70:{torch.mean(voutputs.squeeze()[y2 > 70]):.6f}')
        if epoch % 500 == 0:
            y_pred = voutputs.squeeze().cpu().detach().numpy()
            y_test = y2.cpu().detach().numpy()
            z = np.polyfit(y_test, y_pred, 1)
            p = np.poly1d(z)
            plt.plot(y_test, y_pred, 'o', label='Test data')
            plt.plot(y_test, p(y_test), label=f'Trend: y={z[0]:.3f}x + {z[1]:.3f}')
            plt.title(f"Age Epoch {epoch + 1}")
            plt.legend()
            plt.pause(3)
            plt.show(block=False)
            plt.close()
    y_pred = voutputs.squeeze().cpu().detach().numpy()
    y_test = y2.cpu().detach().numpy()
    z = np.polyfit(y_test, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_test, y_pred, 'o', label='Test data')
    plt.plot(y_test, p(y_test), label=f'Trend: y={z[0]:.3f}x + {z[1]:.3f}')
    plt.title("Age Deep Learning Model")
    plt.legend()
    plt.pause(3)
    plt.savefig('H:/My Drive/Data/classifier/Age_deep4.png')
    plt.show(block=False)
    torch.save(model.state_dict(), "H:/My Drive/Data/classifier/AgeModel4.pth")

def runAgeHuge():
    # get the gpu device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    dataset = CSVDataset("H:/My Drive/Data/classifier/ms8k_211k.csv", "H:/My Drive/Data/classifier/meta.csv", "age")
    # Initialize the k-fold cross validation
    model = nn.Sequential(
        nn.Linear(len(dataset.fheader)-1, 500),
        nn.ReLU(),
        nn.Linear(500, 500),
        nn.ReLU(),
        nn.Linear(500, 500),
        nn.ReLU(),
        nn.Linear(500, 1)
    )
    model = model.float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-7)
    # class_weights = torch.tensor([0.9,0.1])
    # criterion = WeightedMSELoss()
    criterion = nn.MSELoss()
    print("Starting training!")
    dataloader = DataLoader(dataset, batch_size=128)

    epochs = 100
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs.to(device))
            # weights
            # weights = torch.divide(1.0,torch.add(labels,0.1))
            # weights = torch.ones(len(labels))

            # Calculate the loss
            # loss = criterion(outputs.squeeze(), labels.to(device),weights.to(device))
            loss = criterion(outputs.squeeze(), labels.to(device))
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        #voutputs = model(X2.to(device))
        # trainout = model(X.to(device))

        #vloss2 = criterion(voutputs.squeeze(), y2.to(device))
        print(f'Epoch [{epoch + 1}/{epochs}], MSE: {loss.item():.6f}')
        dataset = CSVDataset("H:/My Drive/Data/classifier/ms8k_211k.csv", "H:/My Drive/Data/classifier/meta.csv", "age")
        dataloader = DataLoader(dataset, batch_size=128)
    torch.save(model.state_dict(), "H:/My Drive/Data/classifier/211kAge.pth")


if __name__ == '__main__':
    import matplotlib

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    #runSmoking()
    #runAlcohol()
    #runBMI()
    #runAge()
    #runAge2()
    #runAge3()
    #runAge4()
    runAgeHuge()
