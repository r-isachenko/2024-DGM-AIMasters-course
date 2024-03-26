from collections import defaultdict
from tqdm.notebook import tqdm
from typing import Tuple

import torch
from torch import optim


def train_epoch(
    model: object,
    train_loader: object,
    optimizer: object,
    use_cuda: bool,
    loss_key: str = "total",
    device: str = None,
) -> defaultdict:
    model.train()

    stats = defaultdict(list)
    if device is None:
        if use_cuda:
            device = "cuda"
        else:
            device = "cpu"

    for x in train_loader:
        x = x.to(device)
        losses = model.loss(x)
        optimizer.zero_grad()
        losses[loss_key].backward()
        optimizer.step()

        for k, v in losses.items():
            stats[k].append(v.item())

    return stats


def eval_model(model: object, data_loader: object, 
               use_cuda: bool, device: str = None) -> defaultdict:
    model.eval()
    stats = defaultdict(float)
    if device is None:
        if use_cuda:
            device = "cuda"
        else:
            device = "cpu"

    with torch.no_grad():
        for x in data_loader:
            x = x.to(device)
            losses = model.loss(x)
            for k, v in losses.items():
                stats[k] += v.item() * x.shape[0]

        for k in stats.keys():
            stats[k] /= len(data_loader.dataset)
    return stats


def train_model(
    model: object,
    train_loader: object,
    test_loader: object,
    epochs: int,
    lr: float,
    use_tqdm: bool = False,
    use_cuda: bool = False,
    loss_key: str = "total_loss",
    device: str = None,
) -> Tuple[dict, dict]:
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = defaultdict(list)
    test_losses = defaultdict(list)
    forrange = tqdm(range(epochs)) if use_tqdm else range(epochs)
    if device is None:
        if use_cuda:
            device = "cuda"
        else:
            device = "cpu"
    model = model.to(device)

    for epoch in forrange:
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, use_cuda, loss_key, device=device)
        test_loss = eval_model(model, test_loader, use_cuda, device=device)

        for k in train_loss.keys():
            train_losses[k].extend(train_loss[k])
            test_losses[k].append(test_loss[k])
    return dict(train_losses), dict(test_losses)
