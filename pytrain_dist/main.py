from torch.nn import CrossEntropyLoss
from torchmetrics.text import Perplexity
from transformers import AutoTokenizer

from .config import GlobalConfig
from .data import get_dataloaders
from .model import get_model
from .track_prof import MlTrackContext
from .trainer import Trainer


def run():
    config = GlobalConfig()
    print(config)

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    train_loader, test_loader = get_dataloaders(config, tokenizer=tokenizer)
    model = get_model(config)

    criterion = CrossEntropyLoss(ignore_index=config.pad_token_id)
    metric = Perplexity(ignore_index=config.pad_token_id).to(model.device)

    trainer = Trainer(
        model,
        train_loader,
        test_loader,
        config.epochs,
        config.step_per_epoch,
        config.optimizer_config,
        criterion,
        metric,
        device=model.device,
    )

    with MlTrackContext(config, track=config.track):
        model = trainer.train(
            dev_test=config.dev_test,
            track=config.track,
            profiler_config=config.profiler_config
        )


if __name__ == "__main__":
    run()
