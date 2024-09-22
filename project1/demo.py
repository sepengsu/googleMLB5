from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from transformers import BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch

# IMDb 데이터셋 로드
dataset = load_dataset("imdb")
train_dataset = dataset["train"]
test_dataset = dataset["test"]

print(train_dataset[0])