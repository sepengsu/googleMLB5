import json, os
import zipfile
import tempfile
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import time
import random

class Preprocess:
    def __init__(self, dir_path):
        '''
        dir_path: str, 데이터가 저장된 디렉토리 경로
        dir_path 안에 있는 폴더: Training, Validation
        각각 폴더 안에는 zip 파일로 압축되어 있음
        '''
        self.dir_path = dir_path
        self.train_path = os.path.join(dir_path, 'Training')
        self.val_path = os.path.join(dir_path, 'Validation')
        self.train_data = []
        self.val_data = []

    def __call__(self):
        print('Preprocessing Start')
        self.traindata()
        time.sleep(3)
        print('Train data processing complete')
        self.valdata()
        print('Val data processing complete')
        # 합치기
        data = self.train_data + self.val_data
        # 라벨링
        labeling = Labeling(data)
        data= labeling()
        # 중복데이터 삭제 list 안에 dict로 있음 dict가 같은 경우 중복으로 판단
        data = [dict(t) for t in {tuple(d.items()) for d in data}]
        # Train-test split with randomness
        random.seed(42)
        random.shuffle(data)
        train_size = int(0.8 * len(data))
        train_data = data[:train_size]
        test_data = data[train_size:]
        print('Preprocessing Complete')

        return train_data, test_data, labeling.labels

    def traindata(self):
        path_list = os.listdir(self.train_path)
        path_list = [os.path.join(self.train_path, p) for p in path_list]

        # 병렬 처리 사용: 파일 단위로 처리
        with ProcessPoolExecutor(max_workers=4) as executor:  # max_workers를 명시적으로 설정
            results = list(tqdm(executor.map(self.process_zip, path_list), total=len(path_list), desc="Processing Train Data"))
        
        for result in results:
            self.train_data += result

    def valdata(self):
        path_list = os.listdir(self.val_path)
        path_list = [os.path.join(self.val_path, p) for p in path_list]

        # 병렬 처리 사용: 파일 단위로 처리 (병렬 워커 수를 다르게 설정해볼 수 있음)
        with ProcessPoolExecutor(max_workers=4) as executor:  # val에 대해 워커 수를 줄이거나 조정
            results = list(tqdm(executor.map(self.process_zip, path_list), total=len(path_list), desc="Processing Validation Data"))
        
        for result in results:
            self.val_data += result

    def process_zip(self, zip_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_list = self.unzip(zip_path, temp_dir)
            result = []
            for temp in temp_list:
                result += self.process(temp)
        return result

    def unzip(self, path, extract_to) -> list:
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        paths = os.listdir(extract_to)
        return [os.path.join(extract_to, p) for p in paths]

    def process(self, path) -> list:
        try:
            with open(path, 'r') as f:
                raw = json.load(f)
                raw = raw[0]['sentences']
        except (json.JSONDecodeError, IndexError, KeyError, FileNotFoundError) as e:
            print(f"Error processing file {path}: {e}")
            return []

        data = []
        for s in raw:
            temp = dict()
            temp['text'] = s['origin_text']
            temp['label'] = s['style']['emotion']
            data.append(temp)
        
        return data

class Labeling:
    def __init__(self,data):
        self.labels = []
        self.data = data

    def __call__(self):
        self.labeling()
        return self.data
    
    def labeling(self):
        for d in self.data:
            if d['label'] not in self.labels:
                self.labels.append(d['label'])
            d['label'] = self.labels.index(d['label'])


def saver(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    path = '/home/jaewon/다운로드/133.감성 및 발화 스타일 동시 고려 음성합성 데이터'
    pre = Preprocess(path)
    train_data, test_data, labels = pre()
    labels = {i: l for i, l in enumerate(labels)}
    os.chdir('/home/jaewon/바탕화면/google/Project/project1')
    saver(train_data, './data/train_data.json')
    saver(test_data, './data/test_data.json')
    saver(labels, './data/labels.json')
    print(train_data[0])
