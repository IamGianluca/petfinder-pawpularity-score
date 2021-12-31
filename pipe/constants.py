from pathlib import Path

# paths
path = Path("/workspace")
ckpts_path = path / "ckpts"
data_path = path / "data"
metrics_path = path / "metrics"

train_images_path = data_path / "train"
test_images_path = data_path / "test"
adoption_images_path = data_path / "extra"
dogsvscats_images_path = data_path / "extra2"

# full paths
train_labels_fpath = data_path / "train.csv"
train_deduped_fpath = data_path / "train_deduped.csv"
adoption_labels_fpath = data_path / "extra.csv"
dogsvscats_labels_fpath = data_path / "extra2.csv"

train_5folds_fpath = data_path / "train_5folds.csv"
train_10folds_fpath = data_path / "train_10folds.csv"

cfg_fpath = path / "params.yaml"

# col names
target_col = ["Pawpularity"]
