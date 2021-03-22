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
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
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
        self.learner = tabular_learner(dls = self.dls,
                                       layers = self.LAYERS,
                                       loss_func=MSELossFlat(),
                                       metrics = [SpearmanCorrCoef()])

    def learn(self):
        self.learner.fit_one_cycle(self.N_EPOCHS, self.WEIGHT_DECAY)


    def fit(self, df, **kwargs):
        self.dls = self.build_data_loaders(df, kwargs['cont_names'],
                                           kwargs['train_idx'], kwargs['val_idx'])                                           
        self.init_learner()
        self.learn()
        
    def predict(self, data):
        test_dl = self.dls.test_dl(data)
        preds_out, _ = self.learner.get_preds(dl=test_dl, inner = True)
        preds_out = preds_out.tolist()
        preds_out = [item for sublist in preds_out for item in sublist]
        return preds_out
