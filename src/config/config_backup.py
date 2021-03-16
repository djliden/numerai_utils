from pathlib import Path
from yacs.config import CfgNode as CN
_C = CN()

_C.SYSTEM = CN(new_allowed = True)
_C.TRAIN = CN()
_C.DATA = CN()
_C.MODEL = CN(new_allowed=True)
_C.EVAL = CN()
_C.RESULTS = CN(new_allowed=True)
_C.SESSION = CN(new_allowed=True)
# debugging mode yes or no
_C.SYSTEM.DEBUG = True

# Refresh data
_C.DATA.REFRESH = False
# Save Processed Training Data
_C.DATA.SAVE_PROCESSED_TRAIN = True
_C.DATA.BATCH_SIZE = 1024

_C.TRAIN.N_EPOCHS = 6

_C.EVAL.SAVE_PREDS = True
_C.EVAL.SUBMIT_PREDS = False

_C.MODEL.NAME = "Unnamed"
_C.SESSION.NAME = "Default"


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
