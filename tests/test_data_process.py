import unittest
import pandas as pd
from pathlib import Path
from src.utils import setup
import shutil


TEST_PATH = Path(__file__).parent
DATA_PATH = TEST_PATH / "test_data/"
TMP_PATH = TEST_PATH / "tmp/"


class ProcessData(unittest.TestCase):
    def setUp(self):
        if not (TMP_PATH.exists()):
            TMP_PATH.mkdir()
        self.train = DATA_PATH / "df_train_ex.csv"
        self.tourn = DATA_PATH / "df_val_ex.csv"
        self.processed_pkl = TMP_PATH / "processed.pkl"
        self.training_data, self.feature_cols, self.target_cols = setup.process_current(
            self.processed_pkl, self.train, self.tourn
        )

    def tearDown(self):
        del self.training_data
        del self.feature_cols
        del self.target_cols
        shutil.rmtree(TMP_PATH)

    def test_data_process(self):
        self.assertIsInstance(self.training_data, pd.core.frame.DataFrame)
        self.assertIsInstance(self.feature_cols, pd.core.indexes.base.Index)
        self.assertIsInstance(self.target_cols, list)

        self.assertEquals(len(self.target_cols), 1)
        self.assertEquals(
            len(self.feature_cols),
            sum(self.training_data.columns.str.startswith("feature")),
        )


if __name__ == "__main__":
    unittest.main()

# To execute tests: Use CLI:
# python -m unittest tests.test_data_process
