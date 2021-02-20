import os
import numerapi
from dotenv import load_dotenv, find_dotenv
from getpass import getpass
from pathlib import Path

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
    
    public_key = os.environ.get("NUMERAI_PUBLIC_KEY")
    secret_key = os.environ.get("NUMERAI_SECRET_KEY")
    
    global napi
    napi = numerapi.NumerAPI(verbosity="info", public_id=public_key, secret_key=secret_key)
    print("numerapi initialized as 'napi'")




def download_current(refresh:bool = False, path:str = "./input/"):
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
    