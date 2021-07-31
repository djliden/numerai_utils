import unittest
import pandas as pd
from pathlib import Path
from numerai_utils.src.utils import setup
import shutil


class ProcessData(unittest.TestCase):
    def setUp(self):
        TEST_PATH = Path(__file__).parent
        DATA_PATH = TEST_PATH / "test_data/df_sample.pkl"
        TMP_PATH = TEST_PATH / "tmp/"
        if not TMP_PATH.exists:
            TMP_PATH.mkdir()
        self.df = pd.read_pickle(DATA_PATH)

    def tearDown(self):
        del self.df
        shutil.rmtree(TMP_PATH)

    def test_data_process(self):
        train = TMP_PATH / "tmp_train.pkl"
        tourn = TMP_PATH / "tmp_tourn.pkl"
        processed_pkl = TMP_PATH / "processed.pkl"

        training_data, feature_cols, target_cols = setup.process_current(
            processed_pkl, train, tourn
        )


if __name__ == "__main__":
    unittest.main()
