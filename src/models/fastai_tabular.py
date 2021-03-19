from fastai.tabular.all import *
from pandas.core.frame import DataFrame

class FastaiTabular:
    """Interface for fastai tabular learner

    Keyword arguments:
    df -- Pandas DataFrame with training and validation sets
    train_idx -- list containing indices of training rows
    val_idx -- list containing indices of validation rows

    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def build_data_loaders(self, df, cont_names,
                           train_idx, val_idx):
        splits = (train_idx, val_idx)
        db = TabularPandas(df = df, cat_names = None,
                           cont_names = cont_names,
                           y_names = "target",
                           splits = splits)
        return(db.dataloaders(bs=self.BATCH_SIZE))
    

    def init_learner(self):
        learn = tabular_learner(dls = self.dls,
                                layers = self.LAYERS,
                                loss_func=MSELossFlat(),
                                metrics = [SpearmanCorrCoef()])

    def learn(self):
        self.learn.fit_one_cycle(self.N_EPOCHS, self.WEIGHT_DECAY)


    def fit(self, df, cont_names, train_ix, val_idx):
        self.dls = self.build_data_loaders(df, cont_names,
                                           train_idx, val_idx,
                                           batch_size = self.BATCH_SIZE)
        self.learn = self.init_learner(self.LAYERS)
        self.learn.learn(n_epochs = self.N_EPOCHS, wd = 0)
        
    def predict(self, data):
        self.test_dl = self.dls.test_dl(data)
        preds_out, _ = self.learn.get_preds(dl=test_dl, inner = True)
        preds_out = preds_out.toList()
        preds_out = [item for sublist in preds_out for item in sublist]
        return preds_out
