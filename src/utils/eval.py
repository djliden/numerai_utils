from pathlib import Path
import pandas as pd


def save_preds(
    model, chunksize, pred_path, tourn_path, feature_cols, output=False, save=True
):
    ids = []
    preds = []
    tourn_iter_csv = pd.read_csv(tourn_path, iterator=True, chunksize=chunksize)
    for chunk in tourn_iter_csv:
        df = chunk[feature_cols]
        out = model.predict(df)
        ids.extend(chunk["id"])
        preds.extend(out)
    tourn_iter_csv.close()

    preds_out = pd.DataFrame({"id": ids, "prediction": preds})
    if save:
        if not ((pred_path.parent).exists()):
            pred_path.parent.mkdir()
        preds_out.to_csv(pred_path, index=False)
    if output:
        return preds_out
