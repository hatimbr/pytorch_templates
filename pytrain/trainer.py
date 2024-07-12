import mlflow
import torch
from tqdm import tqdm
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from torch.profiler import profile
from torch.utils.data import DataLoader
from torchmetrics import F1Score

from config import OptimizerConfig, ProfilerConfig
from optimizer import get_optimizer_scheduler
from track_prof import torch_profiler_context


class Trainer:
    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        optimizer_config: OptimizerConfig,
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.optimizer, self.lr_scheduler = get_optimizer_scheduler(
            model,
            total_train_step=len(train_loader) * epochs,
            **optimizer_config.export()
        )
        self.step = 0

    def train_loop(
        self,
        dev_test: bool = False,
        track: bool = False,
        profiler: profile | None = None
    ) -> Tensor:
        self.model.train()
        list_loss = Tensor([]).to("cuda")
        loop = tqdm(self.train_loader, ascii=True)
        criterion = CrossEntropyLoss()

        for i, (input_tensor, pad_mask, sentiments) in enumerate(loop):
            self.optimizer.zero_grad()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            input_tensor = input_tensor.to("cuda")
            pad_mask = pad_mask.to("cuda")
            sentiments = sentiments.to("cuda")

            pred = self.model(input_tensor, pad_mask)
            loss = criterion(pred, sentiments)

            loss.backward()
            self.optimizer.step()

            list_loss = torch.cat((list_loss, loss.detach().data.view(1)))
            avg_loss = list_loss.mean().item()

            if track:
                mlflow.log_metrics(
                    {"loss": loss.item(), "avg_loss": avg_loss},
                    step=self.step,
                )
            loop.set_postfix(loss=loss.item(), avg_loss=avg_loss)
            self.step += 1

            if profiler is not None:
                profiler.step()

            if dev_test and i == 20:
                loop.close()
                print(
                    "Max memory allocated:",
                    torch.cuda.max_memory_allocated(device='cuda')/(1024**3)
                )
                break

        return list_loss

    @torch.no_grad()
    def test_loop(self, dev_test: bool = False, track: bool = False) -> Tensor:
        self.model.eval()
        loop = tqdm(self.test_loader, ascii=True)
        metric = F1Score(
            task="multiclass", num_classes=3, average="weighted"
        ).to("cuda")

        for i, (input_tensor, pad_mask, sentiments) in enumerate(loop):
            input_tensor = input_tensor.to("cuda")
            pad_mask = pad_mask.to("cuda")
            sentiments = sentiments.to("cuda")

            pred = self.model(input_tensor, pad_mask)
            score = metric(pred, sentiments)

            loop.set_postfix(
                it_score=score.cpu().item(), avg_score=metric.compute().cpu().item()
            )

            if dev_test and i == 20:
                loop.close()
                break

        if track:
            mlflow.log_metric(
                "weighted_f1", metric.compute().cpu().item(), step=self.step
            )

        return metric

    def train(
        self,
        epochs: int | None = None,
        dev_test: bool = False,
        track: bool = False,
        profiler_config: ProfilerConfig | None = None
    ) -> Module:
        if epochs is None:
            epochs = self.epochs

        metric = self.test_loop(dev_test=dev_test, track=track)
        print(f"Initial f1 score: {metric.compute().cpu().item()}")

        for epoch in range(epochs):
            print(
                "*"*40, f"Epoch {epoch+1}/{epochs}", "*"*40
            )

            with torch_profiler_context(
                **(None if profiler_config is None else profiler_config.export()),
            ) as profiler:
                list_loss = self.train_loop(
                    dev_test=dev_test, track=track, profiler=profiler
                )

            metric = self.test_loop(dev_test=dev_test, track=track)

            print(
                f"average loss: {list_loss.mean().item()} |",
                f"f1 score: {metric.compute().cpu().item()}"
            )

        return self.model
