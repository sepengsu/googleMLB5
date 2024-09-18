# second.py
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch

# IMDb 데이터셋 로드
dataset = load_dataset("imdb")
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# 모델 및 토크나이저 로드
model_name = "google/gemma-2-2b-it"  # Gemma 2 모델 경로
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# GPU 1을 사용하도록 설정
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)

# 데이터 전처리
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    fp16=True,
    gradient_accumulation_steps=8
)

# Trainer 설정 및 모델 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

# 학습된 모델 저장
model_save_path = "/home/mmai6k_02/workspace/personal_practice/fine_tuned_gemma_sentiment"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"모델이 {model_save_path} 경로에 저장되었습니다.")
