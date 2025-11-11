import torch
from tqdm.auto import tqdm
from sklearn.metrics import f1_score


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    for inputs, targets in tqdm(loader, leave=False, desc='Train'):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        all_preds.append(predicted.cpu())
        all_targets.append(targets.cpu())

    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    f1_macro = f1_score(all_targets, all_preds, average='macro')

    return {
        "loss": running_loss / max(1, total),
        "acc": 100.0 * correct / max(1, total),
        "f1_macro": f1_macro,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, split_name="val"):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    for inputs, targets in tqdm(loader, leave=False, desc=split_name.capitalize()):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        all_preds.append(predicted.cpu())
        all_targets.append(targets.cpu())

    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    f1_macro = f1_score(all_targets, all_preds, average='macro')

    return {
        "loss": running_loss / max(1, total),
        "acc": 100.0 * correct / max(1, total),
        "f1_macro": f1_macro,
    }