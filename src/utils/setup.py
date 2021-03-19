import os
import numerapi
from dotenv import load_dotenv, find_dotenv
from getpass import getpass
from pathlib import Path
import pandas as pd

def credential():
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    if os.getenv("NUMERAI_PUBLIC_KEY"):
        print("Loaded Numerai Public Key into Global Environment!")
    else:
        os.environ["NUMERAI_PUBLIC_KEY"] = getpass("Please enter your Numerai Public Key. You can find your key here: https://numer.ai/submit -> ")
    
    if os.getenv("NUMERAI_SECRET_KEY"):
        print("Loaded Numerai Secret Key into Global Environment!")
    else:
        os.environ["NUMERAI_SECRET_KEY"] = getpass("Please enter your Numerai Secret Key. You can find your key here: https://numer.ai/submit -> ")
    
    
def init_numerapi():
    public_key = os.environ.get("NUMERAI_PUBLIC_KEY")
    secret_key = os.environ.get("NUMERAI_SECRET_KEY")
    napi = numerapi.NumerAPI(verbosity="info", public_id=public_key, secret_key=secret_key)
    return(napi)

def download_current(refresh:bool = False, path:str = "./input/", napi=init_numerapi()):
    "Check if the current dataset exists and download if not"
    data_dir = Path(path)
    round = napi.get_current_round()
    current_file = Path(data_dir/f"numerai_dataset_{round}.zip")

    if not(data_dir.exists()): data_dir.mkdir()
    
    if current_file.exists() and (not refresh):
        print("The dataset has already been downloaded.")
        print("You can re-download it with refresh = True")
    else:
        if current_file.exists(): current_file.unlink()
        napi.download_current_dataset(data_dir)

def process_current(processed_train_file, train_file, tourn_file):
    if processed_train_file.exists():
        print("Loading the processed training data from file\n")
        training_data = pd.read_csv(processed_train_file)
    else:
        tourn_iter_csv = pd.read_csv(tourn_file, iterator=True, chunksize=1e6)
        val_df = pd.concat([chunk[chunk['data_type'] == 'validation'] \
                            for chunk in tqdm(tourn_iter_csv)])
        tourn_iter_csv.close()
        training_data = pd.read_csv(train_file)
        training_data = pd.concat([training_data, val_df])
        training_data.reset_index(drop=True, inplace=True)
        print("Training Dataset Generated! Saving to file ...")
        training_data.to_csv(processed_train_file, index=False)

