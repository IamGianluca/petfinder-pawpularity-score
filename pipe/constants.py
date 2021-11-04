from pathlib import Path

# paths
path = Path("/workspace")
ckpts_path = path / "ckpts"
data_path = path / "data"
metrics_path = path / "metrics"

train_images_path = data_path / "train"
test_images_path = data_path / "test"

# full paths
train_labels_fpath = data_path / "train.csv"
train_folds_fpath = data_path / "train_folds.csv"

cfg_fpath = path / "params.yaml"

# col names
target_col = ["Pawpularity"]
