from config import Config
from data import get_dataloader
from model import get_model

if __name__ == "__main__":
    config = Config(config_file="../config.ini")
    print(config)

    train_loader, test_loader = get_dataloader(config.data_path)
    model = get_model(config.model_path)

    for input_tensor, pad_mask, sentiments in train_loader:
        print(input_tensor.shape, pad_mask.shape, sentiments.shape)
        out = model(input_tensor, pad_mask)
        break
