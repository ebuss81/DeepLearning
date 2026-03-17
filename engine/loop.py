import torch
from tqdm.auto import tqdm
from sklearn.metrics import f1_score


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, save_preds=False):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []
    # optional
    all_logits = []
    all_probs = []

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

        all_preds.append(predicted.detach().cpu())
        all_targets.append(targets.detach().cpu())

        if save_preds:
            all_logits.append(outputs.detach().cpu())
            all_probs.append(torch.softmax(outputs, dim=1).detach().cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    f1_macro = f1_score(all_targets, all_preds, average='macro')

    result = {
        "loss": running_loss / max(1, total),
        "acc": 100.0 * correct / max(1, total),
        "f1_macro": f1_macro,
    }

    if save_preds:
        result["preds"] = all_preds
        result["targets"] = all_targets
        result["logits"] = torch.cat(all_logits)
        result["probs"] = torch.cat(all_probs)

    return result


@torch.no_grad()
def evaluate(model, loader, criterion, device, split_name="val",save_preds=False):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []
    # optional
    all_logits = []
    all_probs = []


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
        if save_preds:
            all_logits.append(outputs.detach().cpu())
            all_probs.append(torch.softmax(outputs, dim=1).detach().cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    f1_macro = f1_score(all_targets, all_preds, average='macro')

    result = {
        "loss": running_loss / max(1, total),
        "acc": 100.0 * correct / max(1, total),
        "f1_macro": f1_macro,
    }

    if save_preds:
        result["preds"] = all_preds
        result["targets"] = all_targets
        result["logits"] = torch.cat(all_logits)
        result["probs"] = torch.cat(all_probs)

    return result
