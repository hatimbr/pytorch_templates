from config import GlobalConfig
from data import get_dataloader
from model import get_model
from trainer import Trainer

if __name__ == "__main__":
    config = GlobalConfig()
    print(config)

    train_loader, test_loader = get_dataloader(config.data_dir)
    model = get_model(config.model_path)

    trainer = Trainer(
        model,
        train_loader,
        test_loader,
        config.epochs,
        config.optimizer_config,
        dev_test=config.dev_test
    )
    trainer.train(epochs=config.epochs)
