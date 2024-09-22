from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from transformers import BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch

# IMDb 데이터셋 로드
dataset = load_dataset("imdb")
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Hugging Face에서 8비트 양자화 적용 설정
bnbConfig = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Hugging Face에서 모델과 토크나이저 불러오기
model_name = "google/gemma-2-2b-it"  # Gemma 2 모델 경로
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2,  # 이진 분류를 위한 레이블 수
    quantization_config=bnbConfig,  # 8비트 양자화 설정 적용
    device_map='auto'  # 가능한 GPU에 자동으로 할당
)

# LoRA 설정 추가
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
)

# LoRA 어댑터를 모델에 추가
model = get_peft_model(model, lora_config)

# GPU 설정
print(f"사용 가능한 GPU 수: {torch.cuda.device_count()}", "이름:", torch.cuda.get_device_name(0))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 캐시 메모리 정리
torch.cuda.empty_cache()

# 데이터 전처리 함수 정의
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=64)

# 데이터 전처리 적용
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# 불필요한 열 제거
train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])

# 데이터셋 포맷을 PyTorch 텐서로 설정
train_dataset.set_format("torch")
test_dataset.set_format("torch")

# 학습 설정 정의
training_args = TrainingArguments(
    output_dir="./results",  # 출력 경로
    eval_strategy="epoch",  # 에포크마다 평가
    learning_rate=2e-5,  # 학습률
    per_device_train_batch_size=8,  # 훈련 배치 크기
    per_device_eval_batch_size=8,  # 평가 배치 크기
    num_train_epochs=3,  # 학습 에포크 수
    weight_decay=0.01,  # 가중치 감쇠
    logging_dir="./logs",  # 로그 디렉토리
    fp16=True,  # 16비트 부동소수점 사용
    gradient_accumulation_steps=1,  # 그래디언트 누적 스텝
    save_steps=1000,  # 저장 스텝
    save_total_limit=2,  # 저장할 체크포인트의 수
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# 모델 파인튜닝 시작
trainer.train()

# 학습된 모델 저장
model_save_path ="./model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"모델이 {model_save_path} 경로에 저장되었습니다.")
