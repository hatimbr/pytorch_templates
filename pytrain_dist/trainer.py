import idr_torch
import mlflow
import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from torch.profiler import profile
from torch.utils.data import DataLoader
from torchmetrics.text import Perplexity
from tqdm import tqdm

from .config import OptimizerConfig, ProfilerConfig
from .optimizer import get_optimizer_scheduler
from .track_prof import TorchProfilerContext


class Trainer:
    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        step_per_epoch: int,
        optimizer_config: OptimizerConfig,
        criterion: CrossEntropyLoss,
        metric: Perplexity,
        device: torch.device,
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.optimizer, self.lr_scheduler = get_optimizer_scheduler(
            model,
            total_train_step=step_per_epoch * epochs,
            **optimizer_config.export()
        )
        self.criterion = criterion
        self.metric = metric
        self.step = 0
        self.step_per_epoch = step_per_epoch
        self.device = device

    def prepare_for_loss(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resize logits and target Tensors for pytorch CrossEntropyLoss"""
        batch_size, seq_length, vocab_size = logits.shape
        logits = logits.view(batch_size * seq_length, vocab_size)
        target = target.view(batch_size * seq_length)
        return logits, target

    def train_loop(
        self,
        dev_test: bool = False,
        track: bool = False,
        profiler: profile | None = None
    ) -> Tensor:
        self.model.train()
        list_loss = torch.Tensor([]).to(self.device)
        loop: tqdm[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = tqdm(
            self.train_loader,
            total=self.step_per_epoch,
            disable=(idr_torch.rank != 0),
            ascii=True,
        )
        for i, (inputs, _, target) in enumerate(loop):
            inputs = inputs.to(self.device)
            target = target.to(self.device)
            out = self.model(input_ids=inputs)

            logits, target = self.prepare_for_loss(out["logits"], target)
            loss: torch.Tensor = self.criterion(logits, target)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # To monitor training
            list_loss = torch.cat((list_loss, loss.detach().data.view(1)))
            avg_loss = list_loss.mean().item()
            # TODO: check if it doesn't slow training
            loop.set_postfix(
                average_loss=avg_loss,
                loss=loss.item(),
                lr=self.optimizer.param_groups[0]["lr"],
            )

            if track:
                mlflow.log_metrics(
                    {
                        "loss": loss.item(),
                        "avg_loss": avg_loss,
                    },
                    step=self.step,
                )

            if profiler:
                self.prof.step()

            if (
                (i == 20 and dev_test)
                or (i == self.step_per_epoch + 1)
            ):
                loop.close()
                print(
                    f"Max memory allocated: \
                    {torch.cuda.max_memory_allocated(device=self.device)/(1024**3)}"
                )
                break

            self.step += 1
        return list_loss

    @torch.no_grad()
    def test_loop(
        self, dev_test: bool = False, track: bool = False
    ) -> Perplexity:
        self.model.eval()
        loop: tqdm[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = tqdm(
            self.test_loader, disable=(idr_torch.rank != 0), ascii=True
        )
        self.metric.reset()

        with torch.no_grad():
            for i, (inp_ids, mask, targets) in enumerate(loop):
                inp_ids = inp_ids.to(self.device)
                mask = mask.to(self.device)
                targets = targets.to(self.device)
                preds = self.model(
                    input_ids=inp_ids, attention_mask=mask
                )

                score: torch.Tensor = self.metric(
                    preds.logits.to(torch.float32), targets
                )
                avg_perplexiy: torch.Tensor = self.metric.compute()

                loop.set_postfix(
                    average_perplexity=avg_perplexiy.item(),
                    perplexity=score.item(),
                )

                if i == 20 and dev_test:
                    loop.close()
                    break

            perplexity = self.metric.compute().item()

            if track:
                mlflow.log_metric("perplexity", perplexity, step=self.step)

        return perplexity

    def train(
        self,
        epochs: int | None = None,
        dev_test: bool = False,
        track: bool = False,
        profiler_config: ProfilerConfig | None = None
    ) -> Module:
        if epochs is None:
            epochs = self.epochs

        perplexity = self.test_loop(dev_test=dev_test, track=track)
        print(f"Initial perplexity score: {perplexity}")

        for epoch in range(epochs):
            print(
                "*"*40, f"Epoch {epoch+1}/{epochs}", "*"*40
            )

            with TorchProfilerContext(
                **({} if profiler_config is None else profiler_config.export()),
            ) as profiler:
                list_loss = self.train_loop(
                    dev_test=dev_test, track=track, profiler=profiler
                )

            perplexity = self.test_loop(dev_test=dev_test, track=track)

            print(
                f"average loss: {list_loss.mean().item()} |",
                f"perplexity score: {perplexity}"
            )

        return self.model
