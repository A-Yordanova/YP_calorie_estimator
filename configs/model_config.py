import timm
from transformers import AutoTokenizer, AutoModel


class MultimodalModelConfig:
    def __init__(self):
        # Text
        self.TEXT_MODEL_NAME = "bert-base-uncased"
        self.TEXT_MODEL = AutoModel.from_pretrained(self.TEXT_MODEL_NAME)
        self.TOKENIZER = AutoTokenizer.from_pretrained(self.TEXT_MODEL_NAME)
        self.TEXT_IN_DIM = self.TEXT_MODEL.config.hidden_size
        self.TEXT_OUT_DIM = 512
        self.TEXT_MODEL_UNFREEZE = "encoder.layer.11|pooler"

        # Images
        self.IMAGE_MODEL_NAME = "resnet50"
        self.IMAGE_MODEL = timm.create_model(
            self.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )
        self.IMAGE_MODEL_CONFIG = timm.get_pretrained_cfg(self.IMAGE_MODEL_NAME)
        self.IMAGE_IN_DIM = self.IMAGE_MODEL.num_features
        self.IMAGE_OUT_DIM = 512
        self.IMAGE_MODEL_UNFREEZE = "layer.3|layer.4"

        # Other features
        self.SCALAR_OUT_DIM = 32
        self.HIDDEN_DIM = 512

        # Data
        self.MASS_MEAN = 216.982571
        self.MASS_STD = 153.920393

        # Training parameters
        self.BATCH_SIZE = 32
        self.TEXT_LR = 2e-5
        self.IMAGE_LR = 1e-5
        self.FUSION_LR = 1e-4
        self.REGRESSOR_LR = 1e-3
        self.EPOCHS = 20
        self.DROPOUT = 0.3
        self.SEED = 42

        # Paths
        self.MODEL_PATH = "model/model.pt"
        self.TRAIN_PATH = "data/train.csv"
        self.VAL_PATH = "data/val.csv"
        self.TEST_PATH = "data/test.csv"
    