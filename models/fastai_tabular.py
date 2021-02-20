# import dependencies
import pandas as pd
import numpy as np
from fastai.tabular.all import *

from ..src.setup import credential
from ..src.setup import download_current

# set flags / seeds
torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
...

# Start with main code
if __name__ == '__main__':
    # Setup Credentials
    credential()
    download_current()