from fvcore.common.registry import Registry
DATASET = Registry('Dataset')


def create_dataset(cfg, subset):
    return DATASET.get(cfg.task.dataset.key)(cfg.task.dataset.info, subset)
