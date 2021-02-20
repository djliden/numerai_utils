# import dependencies
import pandas as pd
import numpy as np
from fastai.tabular.all import *
from pathlib import Path

from src.utils.setup import credential
from src.utils.setup import download_current
from src.utils.setup import init_numerapi

from src.utils.prep_data import get_tabular_pandas_dl

# set flags / seeds
torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# Start with main code
if __name__ == '__main__':
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
    dls = get_tabular_pandas_dl(train=train, tourn=tourn)
