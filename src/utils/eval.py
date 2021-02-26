from pathlib import Path
import numpy as np
import pandas as pd
import os

class FastSubmission:
    '''Class for generating numerai submissions from fastai learners'''

    def __init__(self, dls, learner, filename, 
                 numerapi, chunk:bool = True, chunksize:int = 60000,
                 debug:bool = False, outpath = Path("../output/")):
        self.dls = dls
        self.learn = learner
        self.chunk = chunk
        self.chunksize = chunksize
        self.filename = filename
        self.debug = debug
        self.napi = numerapi
        self.outpath = outpath

    def get_preds_and_ids(self, data_subset):
        data_subset.drop(columns = 'target', inplace = True)
        test_dl = self.dls.test_dl(data_subset)
        preds_out,_ = self.learn.get_preds(dl = test_dl, inner = True)
        preds_out = preds_out.tolist()
        preds_out = [item for sublist in preds_out for item in sublist]

        ids_out = data_subset["id"]
        return(preds_out, ids_out)

    def get_predictions(self, print = False):
        out_list = []
        if self.chunk:
            iter_csv = pd.read_csv(self.filename, iterator=True,
                                   chunksize=self.chunksize)
            out_list.extend(self.get_preds_and_ids(x) for x in iter_csv)
        else:
            pred_data = pd.read_csv(self.filename)
            out_list.extend(self.get_preds_and_ids(pred_data))
        
        preds = []
        ids = []
        preds.extend(x[0] for x in out_list)
        preds_out = [item for sublist in preds for item in sublist]
        ids.extend(x[1] for x in out_list)
        ids_out = [item for sublist in ids for item in sublist]

        predictions_df = pd.DataFrame({
            'id':ids_out,
            'prediction_kazutsugi':preds_out
        }) 

        self.predictions = predictions_df
        if print:
            return(predictions_df)

    def save_predictions(self):
        try:
            self.predictions
        except AttributeError:
            print("No predictions to save.\nRemember to run get_predictions first")
        else:
            print("Saving Predictions...\n")
        if not ((self.outpath).exists()):
            self.outpath.mkdir()
        self.predictions.to_csv(self.outpath/ "predictions.csv",
                                index=False)


    def submit(self):
        try:
            self.predictions
        except AttributeError:
            print("No predictions to submit.\nRemember to run get_predictions first!")
        else:
            if ~((self.outpath / "predictions.csv").exists()):
                self.save_predictions()
                print("Submitting Predictions...\n")
                self.napi.upload_predictions(self.outpath / "predictions.csv",
                                             model_id=os.environ.get("NUMERAI_MODEL_ID"))
