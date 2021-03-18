from fastai.tabular.all import *
from pandas.core.frame import DataFrame

class FastaiTabular:
    """Interface for fastai tabular learner

    Keyword arguments:
    df -- Pandas DataFrame with training and validation sets
    train_idx -- list containing indices of training rows
    val_idx -- list containing indices of validation rows

    """
    
    # def __init__(self, df, train_idx:list, val_idx:list):
    #     self.df = df
    #     self.train_idx = train_idx
    #     self.val_idx = val_idx
    #     self.cont_names = list(self.df.columns\
    #                            [self.df.columns.str.startswith('feature')].values)


    def build_data_loaders(self, batch_size, df, cont_names,
                           train_idx, val_idx):
        splits = (train_idx, val_idx)
        db = TabularPandas(df = df, cat_names = None,
                           cont_names = cont_names,
                           y_names = "target",
                           splits = splits)
        return(db.dataloaders(bs=batch_size))
    

    def init_learner(self, layers):
        learn = tabular_learner(dls = self.dls,
                                layers = layers,
                                loss_func=MSELossFlat(),
                                metrics = [SpearmanCorrCoef()])

    def learn(self, n_epochs, wd):
        self.learn.fit_one_cycle(n_epochs, wd)


    def fit(self, batch_size = 2048, layers = [200, 200]):
        self.dls = self.build_data_loaders(batch_size)
        self.learn = self.init_learner(layers)
        self.learn.learn(n_epochs = 6, wd = 0)
        
    def predict(self, data):
        self.test_dl = self.dls.test_dl(data)
        preds_out, _ = self.learn.get_preds(dl=test_dl, inner = True)
        preds_out = preds_out.toList()
        preds_out = [item for sublist in preds_out for item in sublist]
        return preds_out
    
                                     
