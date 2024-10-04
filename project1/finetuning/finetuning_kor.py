from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from transformers import BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch, os, json
import numpy as np
from evaluate import Evaluator
from evaluate import load
# from history import LossHistoryCallback, save_loss_plot
import pandas as pd

# 1. getcwd() 함수를 사용하여 현재 작업 디렉토리에서 README.md 파일이 있는지 확인, 아니면 exit() 함수를 사용하여 프로그램 종료
if not os.path.exists("README.md"):
    print("README.md 파일이 존재하지 않습니다.")
    exit()

# # 커스텀 데이터셋 불러오기
dataset = load_dataset('json', data_files={'train': './data/train_data.json', 'test': './data/test_data.json'})
dataset = load_dataset('imdb') # 영어 버전으로 대체
train_dataset = dataset['train']
test_dataset = dataset['test']
print(f"훈련 데이터셋 크기: {len(train_dataset)}", f"테스트 데이터셋 크기: {len(test_dataset)}")

# Hugging Face에서 8비트 양자화 적용 설정
bnbConfig = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Hugging Face에서 모델과 토크나이저 불러오기
model_name = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=64)

id2label = {0: "무감정",1: "슬픔",2: "기쁨",3: "분노"}
label2id = {"무감정": 0, "슬픔": 1, "기쁨": 2, "분노": 3}

model =AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels= 4,  # 클래스 수 설정
    quantization_config=bnbConfig,  # 양자화 설정
    device_map={"":0},  # 가능한 GPU에 자동 할당
    id2label=id2label,  # 인덱스를 레이블로 변환 
    label2id=label2id   # 레이블을 인덱스로 변환
)

# LoRA 설정 추가
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=['down_proj', 'gate_proj', 'q_proj', 'o_proj', 'up_proj', 'v_proj', 'k_proj'],
    task_type= "SEQ_CLS"
)

# LoRA 어댑터를 모델에 추가
model = get_peft_model(model, lora_config)

# GPU 설정
print(f"사용 가능한 GPU 수: {torch.cuda.device_count()}", "이름:", torch.cuda.get_device_name(0))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 데이터 전처리 함수 정의
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# 데이터 전처리 적용
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# 훈련 및 테스트 데이터셋을 배치로 변환
train_dataset = train_dataset.with_format("torch")
test_dataset = test_dataset.with_format("torch")

# 불필요한 열 제거 (label 필드를 그대로 사용)
train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])

# 데이터셋 셔플 
train_dataset = train_dataset.shuffle(seed=42)
test_dataset = test_dataset.shuffle(seed=42)

# 훈련 및 테스트 데이터셋을 배치로 변환
train_dataset = train_dataset.with_format("torch")
test_dataset = test_dataset.with_format("torch")


import numpy as np
import evaluate

# accuracy와 f1, precision, recall을 사용하되, 다중 클래스 분류를 위한 'weighted' 평균 방식 적용
acc = evaluate.load("accuracy")
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    # Accuracy는 'average' 매개변수를 사용하지 않음
    accuracy = acc.compute(predictions=predictions, references=labels)

    # F1, Precision, Recall은 'average="weighted"'로 계산
    f1_score = f1.compute(predictions=predictions, references=labels, average="weighted")
    precision_score = precision.compute(predictions=predictions, references=labels, average="weighted")
    recall_score = recall.compute(predictions=predictions, references=labels, average="weighted")

    # 결과를 합쳐서 반환
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1_score["f1"],
        "precision": precision_score["precision"],
        "recall": recall_score["recall"],
    }


def custom_loss(logits, labels):
    # 가중치 계산
    weight_tensor = torch.tensor([0.2032, 0.2704, 0.2529, 0.2735]).to(logits.device).type_as(logits)
    
    # 가중치가 적용된 CrossEntropyLoss 함수 사용
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    
    # 로짓과 레이블로 손실 함수 계산
    loss = loss_fn(logits, labels)
    
    return loss

# output 디렉토리 생성
def create_dir(path):
    # output_dir에 있는 폴더 목록 가져오기
    List = os.listdir(path)
    index = len(List) + 1
    return f"{path}/{path.split('/')[-1]}_{index}"

dir_name = create_dir("./output/kor")

training_args = TrainingArguments(
    output_dir=dir_name,
    eval_steps=500,
    eval_strategy="steps",
    logging_steps=500,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    optim="adamw_8bit",
    weight_decay=0.01,
    logging_dir="./logs",
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    gradient_accumulation_steps=2,
    fp16=False,   # fp16 비활성화
)

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# loss 설정을 위한 Custom Trainer 클래스 정의
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = custom_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss
#Trainer 설정
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,  # 메트릭 함수 추가
    data_collator=data_collator,
)
# 모델 파인튜닝 시작
print(f"모델 파인튜닝을 시작합니다. {dir_name} 경로에 저장됩니다.")
trainer.train()

# 학습된 모델 저장
model_save_path = f"./model/kor/model_{dir_name.split('/')[-1]}"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"모델이 {model_save_path} 경로에 저장되었습니다.")
