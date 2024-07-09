import torch
from pathlib import Path
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
    TransformerEncoderLayer
)
from transformers import Wav2Vec2Model


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


class Classifier(torch.nn.Module):
    def __init__(
        self,
        encoder_hidden_dim=1024,
        nhead=8,
        nb_transformer_encoder_layer=2,
        nb_hidden_layer=1,
        nb_feat_tokens=1,
        nb_class=3,
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

        self.nb_feat_tokens = nb_feat_tokens

    def forward(self, x, pad_mask=None):
        x = self.transformer_encoder(x, src_key_padding_mask=pad_mask)
        x = x[:, 0, :]

        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)

        pred = self.classif(x)
        return pred


class EmoClass(Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.weight = Parameter(torch.randn(encoder.config.num_hidden_layers + 1))

    def forward(self, x: Tensor, pad_mask: Tensor) -> Tensor:
        x = self.encoder(x, attention_mask=pad_mask, output_hidden_states=True)[2]
        x = torch.stack(x, dim=1)

        avg_weight = self.weight / self.weight.sum()
        x = torch.einsum('hijk,i->hjk', x, avg_weight)

        pad_mask = self.encoder._get_feature_vector_attention_mask(
            x.shape[1], pad_mask
        )

        pred = self.classifier(x, pad_mask)

        return pred


def get_model(model_path: Path) -> Module:
    encoder = Wav2Vec2Model.from_pretrained(model_path)
    classifier = Classifier()
    model = EmoClass(encoder, classifier)
    return model
