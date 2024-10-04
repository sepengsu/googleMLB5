from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import os
import numpy as np
from evaluate import load
from transformers import Trainer

# 1. 모델을 불러오기 위해 현재 작업 디렉토리에서 README.md 파일이 있는지 확인, 없으면 프로그램 종료
def test(mode, path):
    if not os.path.exists("README.md"):
        print("README.md 파일이 존재하지 않습니다.")
        exit()
    
    # 데이터셋 불러오기 
    if mode =='eng':
        dataset = load_dataset('imdb')
    else:
        dataset = load_dataset('json', data_files={'train':'./data/train_data.json', 'test':'./data/test_data.json'})
    
    test_dataset = dataset['test']

    # path에 따른 모델 불러오기 
    if path =='default':
        model_name = "google/gemma-2-2b-it"
    else:
        model_name = path
    # model의 num_labels를 모드에 따라 설정
    if mode == 'eng':
        num_labels = 2
    else:
        num_labels = 4
    model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # test 데이터셋 전처리
    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=128)
    
    test_dataset = test_dataset.map(preprocess_function, batched=True)
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    # 100개의 데이터만 사용
    test_dataset = test_dataset
    print(f"테스트 데이터셋 크기: {len(test_dataset)}")
    print(f"라벨 분포: {np.unique(test_dataset['label'], return_counts=True)}")
    # test 진행 (f1, accuracy)
    acc = load('accuracy')
    f1 = load('f1')
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": acc.compute(predictions=predictions, references=labels), "f1": f1.compute(predictions=predictions, references=labels)}
    trainer = Trainer(
        model=model,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    result = trainer.evaluate()
    print(f"Mode: {mode}, path: {path}")
    print(f"accuracy: {result['eval_accuracy']['accuracy']*100: .3f}%, f1: {result['eval_f1']['f1']: .3f}")


def find_path(mode,is_default):
    if not os.path.exists("README.md"):
        print("README.md 파일이 존재하지 않습니다.")
        exit()
    if is_default:
        return 'default'
    else:
        index = os.listdir('./model/'+mode)
        return './model/'+mode+'/'+index[0]
    
if __name__ == '__main__':
    # mode = 'eng'
    # path = find_path(mode,False)
    # test(mode,path)
    # print("="*50)
    # path = find_path(mode,True)
    # test(mode,path)
    print("="*50)
    path = find_path('kor', False)
    print(path)
    test('kor', path)
    



