import torch
from tqdm import tqdm
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics import F1Score


class Trainer:
    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config,
        dev_test: bool = False
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.lr,
        )
        self.dev_test = dev_test

    def train_loop(self) -> Tensor:
        self.model.train()
        list_loss = Tensor([]).to("cuda")
        loop = tqdm(self.train_loader, ascii=True)
        criterion = CrossEntropyLoss()

        for i, (input_tensor, pad_mask, sentiments) in enumerate(loop):
            self.optimizer.zero_grad()

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

            loop.set_postfix(it_score=score, avg_score=metric.compute().cpu().item())

            if self.dev_test and i == 20:
                loop.close()
                break

        return metric

    def train(self, epochs: int) -> Module:
        for epoch in range(epochs):
            print(
                "#"*20, f"Epoch {epoch}/{epochs}", "#"*20
            )

            list_loss = self.train_loop()
            metric = self.test_loop()

            print(
                f"average loss: {list_loss.mean().item()} |",
                f"f1 score: {metric.compute().cpu().item()}"
            )
            print("#" * 60, "\n")

        return self.model
