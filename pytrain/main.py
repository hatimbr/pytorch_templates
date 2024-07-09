from config import Config
from data import get_dataloader
from model import get_model
from trainer import Trainer

if __name__ == "__main__":
    config = Config()
    print(config)

    train_loader, test_loader = get_dataloader(config.data_dir)
    model = get_model(config.model_path)

    trainer = Trainer(model, train_loader, config)
    trainer.train(epochs=1)
