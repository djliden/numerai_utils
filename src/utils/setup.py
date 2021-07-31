import os
import numerapi
from dotenv import load_dotenv, find_dotenv
from getpass import getpass
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import dask.dataframe as dd


def credential():
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    if os.getenv("NUMERAI_PUBLIC_KEY"):
        print("Loaded Numerai Public Key into Global Environment!")
    else:
        os.environ["NUMERAI_PUBLIC_KEY"] = getpass(
            "Please enter your Numerai Public Key. You can find your key here: https://numer.ai/submit -> "
        )

    if os.getenv("NUMERAI_SECRET_KEY"):
        print("Loaded Numerai Secret Key into Global Environment!")
    else:
        os.environ["NUMERAI_SECRET_KEY"] = getpass(
            "Please enter your Numerai Secret Key. You can find your key here: https://numer.ai/submit -> "
        )


def init_numerapi():
    public_key = os.environ.get("NUMERAI_PUBLIC_KEY")
    secret_key = os.environ.get("NUMERAI_SECRET_KEY")
    napi = numerapi.NumerAPI(
        verbosity="info", public_id=public_key, secret_key=secret_key
    )
    return napi


def download_current(
    refresh: bool = False, path: str = "./input/", napi=init_numerapi()
):
    "Check if the current dataset exists and download if not"
    data_dir = Path(path)
    round = napi.get_current_round()
    current_file = Path(data_dir / f"numerai_dataset_{round}.zip")

    if not (data_dir.exists()):
        data_dir.mkdir()

    if current_file.exists() and (not refresh):
        print("The dataset has already been downloaded.")
        print("You can re-download it with refresh = True")
    else:
        if current_file.exists():
            current_file.unlink()
        napi.download_current_dataset(data_dir)


def process_current(
    processed_train_pickle,
    train_file,
    tourn_file,
    pickle_out=True,
):

    if processed_train_pickle.exists():
        print("Loading the pickled training data from file\n")
        training_data = pd.read_pickle(processed_train_pickle)
    else:
        print("Processing training data...\n")
        tourn_iter_csv = dd.read_csv(tourn_file)
        tmp_df = tourn_iter_csv[tourn_iter_csv["data_type"] == "validation"]
        training_data = dd.read_csv(train_file)
        training_data = dd.concat([training_data, tmp_df])
        training_data = training_data.compute()
        training_data.reset_index(drop=True, inplace=True)
        # tourn_iter_csv.close()
        print("Training Dataset Generated! Saving to file ...")

        if pickle_out:
            training_data.to_pickle(processed_train_pickle)

    feature_cols = training_data.columns[
        training_data.columns.str.startswith("feature")
    ]
    target_cols = ["target"]
    return training_data, feature_cols, target_cols
