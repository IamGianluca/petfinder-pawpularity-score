from pytorch_lightning.core.saving import load_hparams_from_yaml


def load_cfg(fpath, cfg_name):
    cfg = load_hparams_from_yaml(config_yaml=fpath, use_omegaconf=True)
    cfg = cfg.get(cfg_name)
    return cfg
