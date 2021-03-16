from fastai.tabular.all import *
from pandas.core.frame import DataFrame

class FastTab:
    """Interface for fastai tabular learner

    Keyword arguments:
    df -- Pandas DataFrame with training and validation sets
    train_idx -- list containing indices of training rows
    val_idx -- list containing indices of validation rows

    """
    
    def __init__(self, df, train_idx:list, val_idx:list):
        self.df = df
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.cont_names = list(self.df.columns\
            [self.df.columns.str.startswith('feature')].values)


    def build_data_loaders(self, batch_size = 2048):
        splits = (self.train_idx, self.val_idx)
        db = TabularPandas(df = self.df, cat_names = None,
                           cont_names = self.cont_names,
                           y_names = "target",
                           splits = splits)
        self.dls = db.dataloaders(bs=batch_size)

    def init_learner(self):
        self.learn = tabular_learner(dls = self.dls,
                                     
