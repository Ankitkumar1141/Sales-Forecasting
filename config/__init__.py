from pathlib import Path
import yaml
import os

CONFIG_FILE_PATH = Path("config/config.yaml")


def read_yaml(path_to_yaml: Path = CONFIG_FILE_PATH) -> dict:
    """
    Reads a YAML configuration file and returns its contents as a dict.
    Returns an empty dict if the file does not exist.
    """
    if not os.path.exists(path_to_yaml):
        return {}
    with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:
        config = yaml.safe_load(yaml_file) or {}
    return config
