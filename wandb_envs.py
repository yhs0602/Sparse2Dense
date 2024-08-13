import os

from dotenv import load_dotenv

load_dotenv()


WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
