import argparse
import configparser
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass(kw_only=True)
class Config:
    config_file: Path = field(
        default="config.ini", metadata={"converter": Path, "track": False}
    )
    lr: float = field(default=1e-05, metadata={"converter": float, "track": True})
    model_dir: Path = field(
        default=Path.cwd() / "model", metadata={"converter": Path, "track": False}
    )
    track: bool = field(default=False, metadata={"converter": bool, "track": False})

    def __post_init__(self) -> None:
        self.fields_names = [field.name for field in fields(self)]

        self.from_file(self.config_file)
        self.from_args()

        self.correct_type()

    def from_file(self, config_path) -> None:
        confparser = configparser.ConfigParser()
        confparser.read(config_path)
        for key, val in confparser["DEFAULT"].items():
            if key in self.fields_names:
                setattr(self, key, val)
            else:
                print(f"Unknown key from config file, ignoring: {key}")

    def from_args(self) -> None:
        argparser = argparse.ArgumentParser()
        for dataclass_field in fields(self):
            argparser.add_argument(
                f"--{dataclass_field.name}",
                action="store_true" if dataclass_field.type == bool else "store",
            )

        args = argparser.parse_args()
        for key, val in vars(args).items():
            if val is not None:
                setattr(self, key, val)

    def correct_type(self) -> None:
        # Convert the values of the config attributes
        for dataclass_field in fields(self):
            converter = dataclass_field.metadata.get("converter", None)
            if converter is not None:
                value = getattr(self, dataclass_field.name)
                if value is not None:
                    self.__setattr__(dataclass_field.name, converter(value))
