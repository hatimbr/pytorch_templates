import torch
from tqdm import tqdm
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from torch.optim import AdamW


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        config
    ):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.lr,
        )

    def train_loop(self) -> Tensor:
        self.model.train()
        list_loss = Tensor([]).to("cuda")
        loop = tqdm(self.train_loader, ascii=True)
        criterion = CrossEntropyLoss()

        for input_tensor, pad_mask, sentiments in loop:
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

        return list_loss

    def train(self, epochs: int) -> Module:
        for epoch in range(epochs):
            print(
                "#"*20, f"Epoch {epoch}/{epochs}", "#"*20
            )

            list_loss = self.train_loop()

            print(f"average loss: {list_loss.mean().item()}")
            print("#" * 60, "\n")

        return self.model
