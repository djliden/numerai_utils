# Try to figure out segfault?
import faulthandler
# import dependencies
import pandas as pd
import numpy as np
from fastai.tabular.all import *
from pathlib import Path

from utils.setup import credential
from utils.setup import download_current
from utils.setup import init_numerapi

from utils.prep_data import get_tabular_pandas_dl

from utils.metrics import sharpe, val_corr
from fastprogress.fastprogress import force_console_behavior


# set flags / seeds
import gc
torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# Start with main code
if __name__ == '__main__':
    faulthandler.enable()
    torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Setup Credentials
    credential()
    napi = init_numerapi()
    download_current(napi = napi)

    # Paths
    round = napi.get_current_round()
    #current_file = Path(data_dir/f"numerai_dataset_{round}.zip")
    train = Path(f"./input/numerai_dataset_{round}/numerai_training_data.csv")
    tourn = Path(f"./input/numerai_dataset_{round}/numerai_tournament_data.csv")

    # Get DataLoaders
    print("setting up fastai dataloaders")
    dls = get_tabular_pandas_dl(train=train, tourn=tourn, refresh=False, save=True)

    # Model Setup
    print("setting up the fastai model")
    learn = tabular_learner(dls, layers=[200,100],
                        loss_func=MSELossFlat(),
                        metrics = [PearsonCorrCoef()])
                        
    master_bar, progress_bar = force_console_behavior()
    # Train Model
    print("training the model")
    learn.fit_one_cycle(2, wd = 2)

    # Get Metrics
    ## Sharpe
    print("Making Predictions on Validation Set")
    prediction, target = learn.get_preds()
    prediction = prediction.numpy().squeeze()
    target = target.numpy().squeeze()
    prediction, target

    era = dls.valid_ds.items['era']
    eval_df = pd.DataFrame({'prediction':prediction, 'target':target, 'era':era}).reset_index()
    sharpe = sharpe(eval_df)

    ## Corr
    correl = val_corr(eval_df)

    print(f'Model training has completed.\nValidation correlation: {correl:.3f}.\nvalidation sharpe: {sharpe:.3f}')

    #print("Saving Predictions")
    del learn
    gc.collect()