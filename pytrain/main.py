import mlflow

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
    )

    if config.track:
        mlflow.set_tracking_uri(config.mlflow_path)
        mlflow.set_experiment(config.exp_name)

        if mlflow.search_runs(filter_string=f"run_name='{config.run_name}'").empty:
            run_id = None
        else:
            run_id = mlflow.search_runs(
                filter_string=f"run_name='{config.run_name}'"
            ).iloc[0]["run_id"]

        with mlflow.start_run(
            run_name=config.run_name, run_id=run_id, log_system_metrics=True
        ) as _:
            mlflow.log_params(config.export())
            model = trainer.train(dev_test=config.dev_test, track=True)

    else:
        model = trainer.train(dev_test=config.dev_test)
