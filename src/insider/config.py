# src/insider/config.py
from dataclasses import dataclass
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

@dataclass(frozen=True)
class Settings:
    data_dir: str = os.getenv("DATA_DIR", "data")
    tiingo_key: str | None = os.getenv("TIINGO_API_KEY")

SETTINGS = Settings()