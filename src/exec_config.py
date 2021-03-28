# import dependencies
import os
import sys
import pandas as pd
import numpy as np
import time
from config.config import Config
from importlib import import_module
from pathlib import Path
from tqdm import tqdm

from utils.eval import save_preds
from utils.cross_val import EraCV
from utils.setup import credential
from utils.setup import download_current
from utils.setup import process_current
from utils.setup import init_numerapi

#import src.models


from utils.metrics import sharpe, val_corr

# set flags / seeds
import gc
#torch.backends.cudnn.benchmark = True
np.random.seed(1)
#torch.manual_seed(1)
#torch.cuda.manual_seed(1)


# Start with main code
if __name__ == '__main__':
    # snake case model argument
    model = sys.argv[1]
    configpath = sys.argv[2]
    default_config = Path("./src/config/default_config.yaml")
    cfg = Config(default_config)
    model_cfg = Path(f'./src/models/default_configs/{model}.yaml')
    model_cls = model.title().replace("_","")
    cfg.update_config(model_cfg)
    cfg.update_config(configpath)
    ct = time.localtime()
    current_time = f'{ct[0]}_{ct[1]}_{ct[2]}_{ct[3]}{ct[4]}'
    cfg.update_config({'SYSTEM':{'TIME':current_time}})
    print(cfg)
    
    # 1. Setup Credentials and NumerAPI object
    credential()
    napi = init_numerapi()

    # 2. Define key paths
    round = napi.get_current_round()
    #current_file = Path(data_dir/f"numerai_dataset_{round}.zip")
    train = Path(f"./input/numerai_dataset_{round}/numerai_training_data.csv")
    tourn = Path(f"./input/numerai_dataset_{round}/numerai_tournament_data.csv")
    processed = Path('./input/training_processed.csv')
    output = Path("./output/")

    # 3. Download and process Data
    download_current(napi = napi)
    #process_current(processed, train, tourn)
    training_data, feature_cols, target_cols = process_current(processed, train, tourn)
    
    # 4. Load Model
    modmod = import_module(f'models.{model}')
    mod = getattr(modmod, model_cls)(**cfg.MODEL.config)

    # CV Setup
    era_split = EraCV(eras = training_data.era)
    corrs = []
    sharpes = []
    X, y, era = training_data[feature_cols], training_data.target, training_data.era
    for valid_era in tqdm(range(cfg.CV.VAL_START, cfg.CV.VAL_END)):
        train_idx, test_idx = era_split.get_splits(valid_start = valid_era,
                                                   valid_n_eras = cfg.CV.VAL_N_ERAS,
                                                   train_start = cfg.CV.TRAIN_START,
                                                   train_stop = cfg.CV.TRAIN_STOP)
        mod.fit(df = training_data, cont_names = list(feature_cols),
                  train_idx = list(train_idx), val_idx = list(test_idx))
        val_preds = mod.predict(X.iloc[test_idx])
        target = list(training_data.target.iloc[test_idx])
        val_era = list(training_data.era[test_idx])
        eval_df = pd.DataFrame({'prediction':val_preds,
                                'target':target, 'era':val_era})
        sharpe_out = sharpe(eval_df)
        corr_out = val_corr(eval_df)
        corrs.append(corr_out)
        sharpes.append(sharpe_out)
    print(f'validation correlations: {corrs}, mean: {np.array(corrs).mean()}')
    print(f'validation sharpes: {sharpes}, mean: {np.array(sharpes).mean()}\n')

    print((f'Model training has completed.\nMean validation correlation:'
           f' {np.array(corrs).mean():.3f}.\nMean validation sharpe: {np.array(sharpes).mean():.3f}'))
    if cfg.EVAL.SAVE_PREDS:
        print(f'Generating and Saving Predictions')
        save_preds(model=mod, chunksize=cfg.EVAL.CHUNK_SIZE,
                   pred_path = output/"predictions/preds.csv",
                   feature_cols = feature_cols,
                   tourn_path=tourn)
        print("Predictions Saved!")
    if cfg.EVAL.SUBMIT_PREDS:
        print("Submitting Predictions!")
        napi.upload_predictions(output/"predictions/preds.csv",
                                model_id=os.environ.get("NUMERAI_MODEL_ID"))

    # Append results to config file
    RESULTS = {'RESULTS':{
        'CORRS':corrs,
        'SHARPES':sharpes,
        'MEAN_CORR':np.array(corrs).mean().item(),
        'MEAN_SHARPE':np.array(sharpes).mean().item()
        }}
    cfg.update_config(RESULTS)
    logname = f'logs/log_{cfg.MODEL.TYPE}_{cfg.MODEL.NAME}_{cfg.SYSTEM.TIME}.yaml'
    cfg.dump_config(path=output/logname)
