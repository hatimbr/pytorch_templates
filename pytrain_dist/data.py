from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import idr_torch
import torch
from torch.utils.data import DataLoader, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizer

from .config import GlobalConfig

RNG = random.Random(53)


class PyDataset(ABC, torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: GlobalConfig,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config

    @abstractmethod
    def make_sampler(self) -> DistributedSampler | None:
        raise NotImplementedError

    def make_dataloader(
        self, *, num_workers: int = 4, prefetch_factor: int = 2
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=self.config.batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            sampler=self.make_sampler(),
            collate_fn=self.collate,
        )

    def collate(
        self, batch: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collate function for the validation data loader.

        It will pad the input and target sequences to the maximum length of the batch.

        Args:
            batch (list): A list of tuples containing input ids and target ids.

        Returns:
            tuple: A tuple containing the input sequence tensor, input mask tensor, and
            target sequence tensor.
        """
        # First remove the last token of the input as well as the first token
        # of the target so that both sentences are shifted by 1 token.
        shifted_batch: list[torch.Tensor] = []
        for sample in batch:
            input_tensor, target_tensor = sample.unbind(dim=-1)
            shifted_input = input_tensor[:-1]
            shifted_target = target_tensor[1:]
            shifted_sample = torch.stack([shifted_input, shifted_target], dim=-1)
            shifted_batch.append(shifted_sample)

        # Now we want to do the padding. rnn.pad_sequence only pads on the right.
        # So we reverse the input, pad the batch and reverse the Tensor (padded batch).
        # The pad value is PAD_TOKEN_ID.
        padded = torch.nn.utils.rnn.pad_sequence(
            [sample.flip(dims=(0,)) for sample in shifted_batch],
            batch_first=True,
            padding_value=self.config.pad_token_id,
        ).flip(dims=(1,))
        input_tensor, target_tensor = padded.unbind(dim=-1)
        # for the training process, the mask should not be useful
        mask = torch.logical_not(
            (input_tensor == torch.full_like(input_tensor, self.config.pad_token_id))
        ).to(dtype=torch.int)
        return input_tensor, mask, target_tensor


class TrainDataset(PyDataset, torch.utils.data.IterableDataset):
    def __iter__(self) -> Iterator[torch.Tensor]:
        iterator = self.infinite_iterator()

        # buffer that will contain the token ids of the current CL_dataset sample
        all_token_ids: list[torch.Tensor] = []
        # buffer that will contain the token ids of the next ticket sample
        next_sample_ids = None
        # index of the current CL_dataset sample, needed for FSDP
        idx = 0
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1
        while True:
            if next_sample_ids is None:
                next_sample_ids = self.get_next_sample(iterator)

            if len(all_token_ids) + len(next_sample_ids) <= self.config.seq_length:
                # if the next HF_dataset sample can fit in the current CL_dataset sample
                # we add it
                all_token_ids += next_sample_ids
                next_sample_ids = None

            else:
                # if the next HF_dataset sample can't fit in the current CL_dataset
                # sample, we add what we can in the CL_dataset sample and then we yield
                # it
                # note: we add one more element compared to seq_length to return to
                # seq_length when generating inputs and targets (see train_collate())
                idx_break = self.config.seq_length - len(all_token_ids)
                all_token_ids += next_sample_ids[: idx_break + 1]
                next_sample_ids = next_sample_ids[idx_break + 1:]

                # yield the sample according to the rank of the dataloader worker
                # and the rank of the distributed training process
                if (
                    idx % (idr_torch.world_size * num_workers)
                    == idr_torch.rank * num_workers + worker_id
                ):
                    yield torch.stack(all_token_ids, dim=0)

                idx += 1
                all_token_ids = []

    def make_sampler(self) -> None:
        return None


class ValidDataset(PyDataset):
    def make_sampler(self) -> DistributedSampler:
        return DistributedSampler(
            self,
            num_replicas=idr_torch.world_size,
            rank=idr_torch.rank,
        )


def _tokenize(text: str, tokenizer: PreTrainedTokenizer) -> list[int]:
    return tokenizer(text, truncation=False, add_special_tokens=False, verbose=False)[
        "input_ids"
    ]


@dataclass
class Chat:
    role: str = field()
    question: str = field()
    generated: list[str] = field()
    nb_answer: int = field()

    @classmethod
    def from_json(cls, json_path: Path) -> Chat:
        chat_dict = json.loads(json_path.read_text())
        return cls(
            role=chat_dict["role"],
            question=chat_dict["question"],
            generated=chat_dict["generated"],
            nb_answer=len(chat_dict["generated"])
        )

    @property
    def input_str(self) -> str:
        return "User: " + self.question + f"\n\n{self.role}:"

    def visual_test(self) -> tuple[str, str]:
        return (
            self.input_str,
            self.generated[RNG.randint(0, self.nb_answer - 1)]
        )

    def tokenize(
        self,
        tokenizer: PreTrainedTokenizer,
        target_only_preds: bool = True,
        pad_token_id: int = 0,
        ans_idx: int | None = None,
    ) -> torch.Tensor:
        if ans_idx is None:
            ans_idx = RNG.randint(0, self.nb_answer - 1)

        input_encoding = _tokenize(self.input_str, tokenizer)
        answer_encoding = _tokenize(self.generated[ans_idx], tokenizer)
        if target_only_preds:
            target_encoding = [pad_token_id] * len(input_encoding)
            input_encoding += answer_encoding
            target_encoding += answer_encoding
        else:
            input_encoding += answer_encoding
            target_encoding = input_encoding.copy()
        if tokenizer.bos_token_id is not None:
            input_encoding.insert(0, tokenizer.bos_token_id)
            target_encoding.insert(0, tokenizer.bos_token_id)
        input_encoding.append(tokenizer.eos_token_id)
        target_encoding.append(tokenizer.eos_token_id)
        input_tensor = torch.tensor(input_encoding, dtype=torch.int64)
        target_tensor = torch.tensor(target_encoding, dtype=torch.int64)
        return torch.stack([input_tensor, target_tensor], dim=-1)


class TrainRoleDataset(TrainDataset, torch.utils.data.IterableDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: GlobalConfig,
        data_path: Path,
    ):
        super().__init__(tokenizer, config)
        self.data_path = data_path
        self.rng = random.Random(53)

    def infinite_iterator(self) -> Iterator[Path]:
        files = list(self.data_path.glob(r"*/*.json"))
        while True:
            self.rng.shuffle(files)
            yield from files

    def get_next_sample(self, iterator: Iterator[Path]) -> torch.Tensor:
        """Get the next sample in the files list"""
        json_path = next(iterator)
        chat = Chat.from_json(json_path)
        return chat.tokenize(
            tokenizer=self.tokenizer,
            target_only_preds=False,
            pad_token_id=self.config.pad_token_id,
        )


class ValidRoleDataset(ValidDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: GlobalConfig,
        data_path
    ) -> None:
        super().__init__(tokenizer, config)
        self.json_list = list(data_path.glob(r"*/*.json"))

    def __len__(self) -> int:
        return len(self.json_list)

    def visual_test(self, nb_sample: int = 2) -> tuple[list[str], torch.Tensor, list[str]]:
        sampled_jsons = RNG.sample(self.json_list, nb_sample)
        inp_prompts = []
        label_prompts = []
        for json_path in sampled_jsons:
            inp_prompt, label_prompt = Chat.from_json(json_path).visual_test()
            inp_prompts.append(inp_prompt)
            label_prompts.append(label_prompt)

        inp_tensor = self.tokenizer(inp_prompts, return_tensors="pt", padding=True)
        return inp_prompts, inp_tensor, label_prompts

    def __getitem__(self, idx: int) -> torch.Tensor:
        chat = Chat.from_json(self.json_list[idx])

        return chat.tokenize(
            tokenizer=self.tokenizer,
            target_only_preds=False,
            pad_token_id=self.config.pad_token_id,
        )


def get_dataloaders(
    config: GlobalConfig, tokenizer: PreTrainedTokenizer,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = TrainRoleDataset(
        tokenizer, config=config, data_path=config.data_dir / "train",
    )
    valid_dataset = ValidRoleDataset(
        tokenizer, config=config, data_path=config.data_dir / "test",
    )

    dataloader_kwargs = {
        "num_workers": 1,
        "prefetch_factor": 2,
    }
    train_loader = train_dataset.make_dataloader(**dataloader_kwargs)
    valid_loader = valid_dataset.make_dataloader(**dataloader_kwargs)
    return train_loader, valid_loader
