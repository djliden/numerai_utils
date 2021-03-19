# import dependencies
import sys
import pandas as pd
import numpy as np
import time
from config.config import Config
from importlib import import_module
from pathlib import Path

from utils.eval import FastSubmission
from utils.setup import credential
from utils.setup import download_current
from utils.setup import process_current
from utils.setup import init_numerapi

#import src.models

from src.utils.prep_data import get_tabular_pandas_dl

from src.utils.metrics import sharpe, val_corr

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
    default_config = Path("./config/default_config.yaml")
    cfg = Config(default_config)
    model_cfg = Path(f'./models/default_configs/{model}.yaml')
    model_cls = model.title().replace("_","")
    cfg.update_config(model_cfg)
    cfg.update_config(configpath)
    ct = time.localtime()
    current_time = f'{ct[0]}_{ct[1]}_{ct[2]}_{ct[3]}{ct[4]}'
    cfg.update_config({'SYSTEM':{'TIME':current_time}})
    print(cfg)
    
    torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    process_current(processed, train, tourn)
    
    # 4. Load Model
    modmod = import_module(f'src.models.{model}')
    mod = getattr(modmod, model_cls)(cfg.MODEL.config)

    # CV Setup
    
    
    # Get DataLoaders
    print("setting up fastai dataloaders")
    dls = get_tabular_pandas_dl(train=train, tourn=tourn,
                                refresh=cfg.DATA.REFRESH,
                                save=cfg.DATA.SAVE_PROCESSED_TRAIN,
                                debug = cfg.SYSTEM.DEBUG,
                                batchsize = cfg.DATA.BATCH_SIZE)

    # Model Setup
    print("setting up the fastai model")
    mod_config = tabular_config(ps=cfg.MODEL.DROPOUT_P, # per-layer dropout prob
                                embed_p=cfg.MODEL.EMBED_DROPOUT_P,
                                y_range=cfg.MODEL.Y_RANGE,
                                use_bn=cfg.MODEL.USE_BATCHNORM, #use batchnorm
                                bn_final=cfg.MODEL.BATCHNORM_FINAL,
                                bn_cont=True, #batchnorm continuous vars
                                )
    learn = tabular_learner(dls, layers=cfg.MODEL.LAYERS,
                            loss_func=MSELossFlat(),
                            lr = cfg.MODEL.LEARNING_RATE,
                            metrics = [PearsonCorrCoef()],
                            config = mod_config)
                        
    #master_bar, progress_bar = force_console_behavior()
    # Train Model
    print("training the model\n")
    print(("[N | train --------- | valid ----------- |"
           " corr ------------- | time]"))
    start = time.time()
    with learn.no_bar():
        learn.fit_one_cycle(n_epoch = cfg.TRAIN.N_EPOCHS,
                            wd = cfg.MODEL.WEIGHT_DECAY)
    end = time. time()

    # Get Metrics
    ## Sharpe
    print("Making Predictions on Validation Set")
    with learn.no_bar():
        prediction, target = learn.get_preds()
    
    prediction = prediction.numpy().squeeze()
    target = target.numpy().squeeze()
    prediction, target

    era = dls.valid_ds.items['era']
    eval_df = pd.DataFrame({'prediction':prediction,
                            'target':target, 'era':era}).reset_index()
    sharpe = sharpe(eval_df)

    ## Corr
    correl = val_corr(eval_df)

    print((f'Model training has completed.\nValidation correlation:'
           f' {correl:.3f}.\nvalidation sharpe: {sharpe:.3f}'))
    if cfg.EVAL.SAVE_PREDS:
        print(f'Generating and Saving Predictions')
        predictions = FastSubmission(dls = dls, learner=learn, chunk=True,
                                     chunksize = 100000, numerapi = napi,
                                     filename = tourn)
        print("Generating Predictions...\n")
        with learn.no_bar():
            predictions.get_predictions(print=False)
        predictions.save_predictions()
        print("Predictions Saved!")
    else:
        print("Following configuration: not saving predictions")
    if cfg.EVAL.SUBMIT_PREDS:
        predictions.submit()

    # Append results to config file
    cfg.defrost()
    cfg.RESULTS.CORREL = correl
    cfg.RESULTS.SHARPE = sharpe
    cfg.RESULTS.TIME = end-start
    #cfg.merge_from_list(results)
    # Export Config+Results as Log
    log_name = (f'logs/'
                f'{cfg.MODEL.GROUP}_'
                f'{cfg.MODEL.NAME}_'
                f'{cfg.SYSTEM.TIME}'
                f'.yaml'
                )
    cfg.dump(stream=open(output / log_name, 'w'))

    del learn
    gc.collect()
