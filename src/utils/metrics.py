from scipy.stats import spearmanr
import pandas as pd
import numpy as np


def sharpe(df: pd.DataFrame) -> np.float32:
    """
    Calculate the Sharpe ratio by using grouped per-era data
    :param df: A Pandas DataFrame containing the columns "era", "target" and "prediction"
    :return: The Sharpe ratio for your predictions.
    """

    def _score(sub_df: pd.DataFrame) -> np.float32:
        """Calculate Spearman correlation for Pandas' apply method"""
        return spearmanr(sub_df["target"], sub_df["prediction"])[0]

    corrs = df.groupby("era").apply(_score)
    return corrs.mean() / corrs.std()


def val_corr(df: pd.DataFrame) -> np.float32:
    """
    Calculate the correlation by using grouped per-era data
    :param df: A Pandas DataFrame containing the columns "era", "target" and "prediction"
    :return: The average per-era correlations.
    """

    def _score(sub_df: pd.DataFrame) -> np.float32:
        """Calculate Spearman correlation for Pandas' apply method"""
        return spearmanr(sub_df["target"], sub_df["prediction"])[0]

    corrs = df.groupby("era").apply(_score)
    return corrs.mean()

def sharpe_s(preds: list, target: list) -> np.float32:
    """Calculate the Sharpe ratio by using grouped per-era data

    >>> sharpe_s([1,2,3,4,5], [1,2,4,5,3])
    0.7
    """
    return spearmanr(target, preds)[0]


if __name__ == '__main__':
    import doctest
    doctest.testmod()
