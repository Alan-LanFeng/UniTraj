from .autobot_dataset import AutoBotDataset
from .wayformer_dataset import WayformerDataset
from .MTR_dataset import MTRDataset

__all__ = {
    'autobot': AutoBotDataset,
    'wayformer': WayformerDataset,
    'MTR': MTRDataset,
}

def build_dataset(config,val=False):
    dataset = __all__[config.method.model_name](
        config=config, is_validation=val
    )
    return dataset
