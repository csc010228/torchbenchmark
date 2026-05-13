from torchbenchmark.tasks import COMPUTER_VISION
from torchbenchmark.util.framework.vision.model_factory import TorchVisionModel
from torchvision import models


class Model(TorchVisionModel):
    task = COMPUTER_VISION.CLASSIFICATION

    # Same as VGG16
    DEFAULT_TRAIN_BSIZE = 64
    DEFAULT_EVAL_BSIZE = 4

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(
            model_name="vgg19",
            test=test,
            device=device,
            batch_size=batch_size,
            weights=models.VGG19_Weights.IMAGENET1K_V1,
            extra_args=extra_args,
        )
