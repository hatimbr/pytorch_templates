from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class FriendsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_path: Path):
        self.df = df
        self.sentiments_dict = {
            "neutral": 0,
            "positive": 1,
            "negative": 2,
        }
        self.data_path = data_path

    def __len__(self) -> int:
        """Return the number of element of the dataset"""
        return len(self.df)

    def __getitem__(self, idx) -> tuple[torch.Tensor, int, int, int]:
        """Return the input for the model and the label for the loss"""
        df_elem = self.df.loc[idx]

        sentiment = self.sentiments_dict[df_elem['Sentiment']]

        audio_torch = torch.load(
            self.data_path /
            f"dia{df_elem['Dialogue_ID']}_utt{df_elem['Utterance_ID']}.pt"
        )

        # Truncate the audio to 400000 samples to avoir cuda oom
        if audio_torch.shape[0] > 400000:
            audio_torch = audio_torch[:400000]

        return audio_torch, sentiment


def collate_fn(batch):
    audios_torch_list = [element[0] for element in batch]
    sentiments = torch.tensor([element[1] for element in batch], dtype=torch.int64)

    input_tensor = torch.nn.utils.rnn.pad_sequence(
        audios_torch_list,
        batch_first=True,
        padding_value=0.0,
    )
    pad_mask = (input_tensor != 0.).to(dtype=torch.int)

    return input_tensor, pad_mask, sentiments


def get_dataloader(data_path: Path) -> tuple[DataLoader, DataLoader]:
    train_df = pd.read_csv(data_path / "train_sent_emo.csv")
    test_df = pd.read_csv(data_path / "test_sent_emo.csv")

    train_dataset = FriendsDataset(train_df, data_path / "train_pt")
    test_dataset = FriendsDataset(test_df, data_path / "test_pt")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        num_workers=1,
        prefetch_factor=1,
        shuffle=True,
        collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=4,
        num_workers=1,
        prefetch_factor=1,
        collate_fn=collate_fn
    )

    return train_dataloader, test_dataloader
