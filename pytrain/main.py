from .config import GlobalConfig
from .data import get_dataloader
from .model import get_model
from .track_prof import MlTrackContext
from .trainer import Trainer


def run():
    config = GlobalConfig()
    print(config)

    train_loader, test_loader = get_dataloader(config.dataset_path, config.batch_size)
    model = get_model(
        config.model_config.encoder_path, **config.model_config.export("kwargs")
    )

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


if __name__ == "__main__":
    run()
