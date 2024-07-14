from pathlib import Path

import torch
from torch import Tensor
from torch.nn import (
    Dropout,
    Linear,
    Module,
    ModuleList,
    Parameter,
    ReLU,
    Sequential,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from transformers import Wav2Vec2Model


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


class Classifier(torch.nn.Module):
    def __init__(
        self,
        encoder_hidden_dim: int = 1024,
        nhead: int = 8,
        nb_transformer_encoder_layer: int = 2,
        nb_hidden_layer: int = 1,
        nb_class: int = 3,
    ):
        super(Classifier, self).__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model=encoder_hidden_dim,
            nhead=nhead,
            activation="gelu",
            norm_first=True,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_layers=nb_transformer_encoder_layer
        )

        self.hidden_layers = ModuleList([
            Sequential(
                Linear(encoder_hidden_dim, encoder_hidden_dim),
                ReLU(),
                Dropout(0.1)
            )
            for _ in range(nb_hidden_layer)
        ])

        self.classif = Linear(encoder_hidden_dim, nb_class)

    def forward(self, x, pad_mask=None):
        x = self.transformer_encoder(x, src_key_padding_mask=pad_mask)
        x = x[:, 0, :]

        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)

        pred = self.classif(x)
        return pred


class EmoClass(Module):
    def __init__(
        self,
        encoder: Wav2Vec2Model,
        nhead: int = 8,
        nb_transformer_encoder_layer: int = 2,
        nb_hidden_layer: int = 1,
        nb_class: int = 3,
    ):
        super().__init__()
        self.encoder = encoder
        self.classifier = Classifier(
            encoder_hidden_dim=encoder.config.hidden_size,
            nhead=nhead,
            nb_transformer_encoder_layer=nb_transformer_encoder_layer,
            nb_hidden_layer=nb_hidden_layer,
            nb_class=nb_class,
        )
        self.weight = Parameter(torch.randn(encoder.config.num_hidden_layers + 1))

    def forward(self, x: Tensor, pad_mask: Tensor) -> Tensor:
        x = self.encoder(x, attention_mask=pad_mask, output_hidden_states=True)[2]
        x = torch.stack(x, dim=1)

        avg_weight = self.weight / self.weight.sum()
        x = torch.einsum('hijk,i->hjk', x, avg_weight)

        pad_mask = ~self.encoder._get_feature_vector_attention_mask(
            x.shape[1], pad_mask
        )

        pred = self.classifier(x, pad_mask)

        return pred


def get_model(encoder_path: Path, **kwargs) -> Module:
    encoder = freeze_model(Wav2Vec2Model.from_pretrained(encoder_path))
    model = EmoClass(encoder, **kwargs).to("cuda")
    print(model)
    return model
