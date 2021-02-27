from os import PathLike
import pandas as pd
from tqdm import tqdm
from fastai.tabular.all import *

def get_tabular_pandas_dl(train:PathLike, tourn:PathLike, debug:bool,
                          use_era:bool = False, save = True,
                          refresh = False):
    data_dir = train.parents[0]

    if refresh or not((data_dir/"training_processed.csv").exists()):
        print("Reading tournament data in chunks")
        iter_csv = pd.read_csv(tourn, iterator=True, chunksize=1e5)
        val_df = pd.concat([chunk[chunk['data_type'] == 'validation'] for chunk in tqdm(iter_csv)])
        iter_csv.close()

        training_data = pd.read_csv(train)
        training_data = pd.concat([training_data, val_df])
        training_data.reset_index(drop=True, inplace=True)

        if save:
            "Writing the processed training data to file\n"
            training_data.to_csv((data_dir / "training_processed.csv"), index=False)

    if (data_dir / "training_processed.csv").exists():
        print("Loading the processed training data from file\n")
        training_data = pd.read_csv(data_dir / "training_processed.csv")

    if (debug):
        print("using a debugging sample of 1500 rows\n")
        training_data = training_data.sample(1500)
        training_data.reset_index(drop=True, inplace=True)

    feature_cols = training_data.columns[training_data.columns.str.startswith('feature')]
    target_cols = ['target']

    train_idx = training_data.index[training_data.data_type=='train'].tolist()
    test_idx = training_data.index[training_data.data_type=='validation'].tolist()
    
    splits = (list(train_idx), list(test_idx))
    categorical = ['era'] if use_era else None

    data = TabularPandas(training_data, cat_names=categorical,
                        cont_names=list(feature_cols.values),
                        y_names=target_cols, splits = splits)

    return(data.dataloaders())
