import torch


def collate_fn(batch):
    batch = list(zip(*batch))
    for i in range(len(batch)):
        if isinstance(batch[i][0], torch.Tensor):
            batch[i] = torch.stack(batch[i])
    return tuple(batch)
