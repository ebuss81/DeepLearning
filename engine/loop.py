import torch
from tqdm.auto import tqdm
from sklearn.metrics import f1_score


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    for inputs, targets in tqdm(loader, leave=False, desc='Train'):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=False):#(device == "cuda")):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        all_preds.append(predicted.cpu())
        all_targets.append(targets.cpu())

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
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=False):#(device == "cuda")):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        running_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        all_preds.append(predicted.cpu())
        all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    f1_macro = f1_score(all_targets, all_preds, average='macro')

    return {
        "loss": running_loss / max(1, total),
        "acc": 100.0 * correct / max(1, total),
        "f1_macro": f1_macro,
    }
