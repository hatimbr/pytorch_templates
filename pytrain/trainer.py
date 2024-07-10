import torch
from tqdm import tqdm
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import DataLoader
from torchmetrics import F1Score

from config import OptimizerConfig
from optimizer import get_optimizer_scheduler


class Trainer:
    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        optimizer_config: OptimizerConfig,
        dev_test: bool = False
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.optimizer, self.lr_scheduler = get_optimizer_scheduler(
            model,
            optimizer_config.optimizer_name,
            optimizer_config.lr,
            optimizer_config.beta1,
            optimizer_config.beta2,
            optimizer_config.eps,
            optimizer_config.weight_decay,
            optimizer_config.momentum,
            optimizer_config.lr_scheduler_name,
            total_train_step=len(train_loader) * epochs,
            num_warmup_steps=optimizer_config.num_warmup_steps
        )
        self.dev_test = dev_test

    def train_loop(self) -> Tensor:
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

            loop.set_postfix(loss=loss.item(), avg_loss=avg_loss)

            if self.dev_test and i == 20:
                loop.close()
                print(
                    "Max memory allocated:",
                    torch.cuda.max_memory_allocated(device='cuda')/(1024**3)
                )
                break

        return list_loss

    @torch.no_grad()
    def test_loop(self) -> Tensor:
        self.model.eval()
        loop = tqdm(self.test_loader, ascii=True)
        metric = F1Score(task="multiclass", num_classes=3).to("cuda")

        for i, (input_tensor, pad_mask, sentiments) in enumerate(loop):
            input_tensor = input_tensor.to("cuda")
            pad_mask = pad_mask.to("cuda")
            sentiments = sentiments.to("cuda")

            pred = self.model(input_tensor, pad_mask)
            score = metric(pred, sentiments)

            loop.set_postfix(
                it_score=score.cpu().item(), avg_score=metric.compute().cpu().item()
            )

            if self.dev_test and i == 20:
                loop.close()
                break

        return metric

    def train(self, epochs: int = None) -> Module:
        if epochs is None:
            epochs = self.epochs

        for epoch in range(epochs):
            print(
                "*"*40, f"Epoch {epoch+1}/{epochs}", "*"*40
            )

            list_loss = self.train_loop()
            metric = self.test_loop()

            print(
                f"average loss: {list_loss.mean().item()} |",
                f"f1 score: {metric.compute().cpu().item()}"
            )

        return self.model
