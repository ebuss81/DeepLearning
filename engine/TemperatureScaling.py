import torch
import torch.nn as nn
import torch.nn.functional as F

class TempScaledModel(nn.Module):
    """
    Wraps a logits-producing model and applies a learned scalar temperature T.
    Use .fit_temperature(val_loader) once after training the base model.
    """
    def __init__(self, model, init_T=1.0, device=None):
        super().__init__()
        self.model = model
        # We optimize log_T for positivity, T = softplus(log_T) + eps
        self.log_T = nn.Parameter(torch.log(torch.exp(torch.tensor(init_T)) - 1.0))
        self.eps = 1e-6
        self._device = device

    @property
    def T(self):
        # strictly positive scalar
        return F.softplus(self.log_T) + self.eps

    def forward(self, x):
        logits = self.model(x)           # (B, C) or (B,) for binary with single logit
        T = self.T
        # If binary single-logit, make it (B,2) for consistency only when you need probs
        return logits / T

    @torch.no_grad()
    def predict_proba(self, x):
        logits_T = self.forward(x)
        if logits_T.ndim == 1 or logits_T.shape[-1] == 1:     # binary
            return torch.sigmoid(logits_T).unsqueeze(-1)      # (B,1) probability of class 1
        return torch.softmax(logits_T, dim=-1)

    def fit_temperature(self, val_loader, max_epochs=200, lr=0.05, binary=False):
        """
        Optimize T on a calibration set to minimize NLL.
        - val_loader must yield (x, y) with y as class indices for multiclass
          or {0,1} for binary.
        """
        self.train(False)
        device = self._device or next(self.parameters()).device
        opt = torch.optim.LBFGS([self.log_T], lr=0.5, max_iter=max_epochs, line_search_fn='strong_wolfe')

        # Pre-collect logits and labels for speed (and to detach from base model)
        all_logits, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = self.model(xb)        # unscaled logits
                all_logits.append(logits.detach())
                all_targets.append(yb.detach())
        logits = torch.cat(all_logits, dim=0)
        targets = torch.cat(all_targets, dim=0)

        # Handle shapes: for binary BCE with single logit
        is_binary_single_logit = (logits.ndim == 1) or (logits.shape[-1] == 1)
        if is_binary_single_logit:
            logits = logits.view(-1)  # (N,)

        nll = nn.CrossEntropyLoss()   # for multiclass
        bce = nn.BCEWithLogitsLoss()  # for binary

        def closure():
            opt.zero_grad()
            T = self.T
            if is_binary_single_logit:
                loss = bce(logits / T, targets.float())
            else:
                loss = nll(logits / T, targets.long())
            loss.backward()
            return loss

        opt.step(closure)

        # Optional: report final temperature & NLL
        with torch.no_grad():
            T = float(self.T.detach().cpu())
            if is_binary_single_logit:
                before = bce(logits, targets.float()).item()
                after  = bce(logits / T, targets.float()).item()
            else:
                before = nll(logits, targets.long()).item()
                after  = nll(logits / T, targets.long()).item()
        print(f"[TempScaling] Learned T = {T:.4f} | NLL before={before:.4f}, after={after:.4f}")

        return T