import numpy as np
import pandas as pd
from prefetch_generator import BackgroundGenerator
from scipy.stats import spearmanr
from tqdm.notebook import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class NumerTorch(nn.Module):
    def __init__(self, n_features, layer_sizes):
        super(NumerTorch, self).__init__()
        layers = [nn.Linear(n_features, layer_sizes[0]),
        nn.ReLU()]

        for i in range(0, len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(layer_sizes[-1], 1))

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.net(x)
        return out

class NumerData(Dataset):
    def __init__(self, data, feature_cols, target_cols):
        self.data = data
        self.features = data[feature_cols].copy().values.astype(np.float32)
        self.targets = data[target_cols].copy().values.astype(np.float32)
        self.eras = data.era.copy().values

    def __len__(self):
        return(len(self.data))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist() 

        return self.features[idx], self.targets[idx], self.eras[idx]

def era_spearman(preds:float, targs:float, eras:str) -> np.float32:
    """
    Calculate the correlation by using grouped per-era data
    :param preds: A list or array of predictions
    :param actual: A list or array of true values
    :param eras: A list or array of eras for grouping
    :return: The average per-era correlations.
    """
    df = pd.DataFrame({"target": targs,
                       "prediction": preds,
                       "era": eras})
    def _score(sub_df: pd.DataFrame) -> np.float32:
        """ Calculate Spearman correlation for Pandas' apply method """
        return spearmanr(sub_df["target"],  sub_df["prediction"])[0]
    corrs = df.groupby("era").apply(_score)
    return corrs.mean() 

def train(epoch, model, train_dl, optim, criterion):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    # set up tqdm bar
    pbar = tqdm(enumerate(BackgroundGenerator(train_dl)),
                total=len(train_dl), position=0, leave=False)
    
    for batch_idx, (data, target, era) in pbar:
        data, target = data.to(device), target.to(device)

        # reset gradients
        optim.zero_grad()
        
        # forward pass
        out = model(data)

        #compute loss
        loss = criterion(out, target)

        l1_lambda = 0.0001
        l1_norm = sum(p.abs().sum()
                      for p in model.parameters())
        loss = loss + l1_lambda * l1_norm

        #backpropagation
        loss.backward()
        
        #update the parameters
        optim.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch/Batch: {epoch}/{batch_idx}\tTraining Loss: {loss.item():.4f}')


def test(model, val_dl, criterion):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target, era in val_dl:
            data, target = data.to(device), target.to(device)
            
            out = model(data)
            test_loss += criterion(out, target).item() # sum up batch loss
            val_corr = era_spearman(preds = out.cpu().numpy().squeeze(),
                                    targs = target.cpu().numpy().squeeze(),
                                    eras = era)

        #test_loss /= len(val_dl.dataset)
        print(f'Test Loss: {test_loss:.4f}, Test Correlation: {val_corr:.4f}')