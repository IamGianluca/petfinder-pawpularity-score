import torchmetrics


def metric_factory(name):
    if name == "auc":
        return torchmetrics.AUROC(pos_label=1)
    elif name == "mse":
        return torchmetrics.MeanSquaredError()
    else:
        raise ValueError("Metric not supported yet.")
