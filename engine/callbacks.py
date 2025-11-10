class EarlyStopping:
    """
    Early stopping on a metric (e.g., val_acc).

    mode = "max": higher is better
    mode = "min": lower is better
    """

    def __init__(self, patience=50, mode="max"):
        assert mode in ["max", "min"]
        self.patience = patience
        self.mode = mode
        self.best = None
        self.num_bad_epochs = 0

    def step(self, metric, epoch=None):
        if self.best is None:
            self.best = metric
            self.num_bad_epochs = 0
            return False

        improve = (
            metric > self.best if self.mode == "max" else metric < self.best
        )
        if improve:
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True
        return False
