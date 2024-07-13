from config import GlobalConfig
from data import get_dataloader
from model import get_model
from track_prof import MlTrackContext
from trainer import Trainer

if __name__ == "__main__":
    config = GlobalConfig()
    print(config)

    train_loader, test_loader = get_dataloader(config.dataset_path)
    model = get_model(config.model_path)

    trainer = Trainer(
        model,
        train_loader,
        test_loader,
        config.epochs,
        config.optimizer_config,
    )

    with MlTrackContext(config, track=config.track):
        model = trainer.train(
            dev_test=config.dev_test,
            track=config.track,
            profiler_config=config.profiler_config
        )
