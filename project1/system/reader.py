import os
import numpy as np

class ModelReader:
    def __init__(self, mode, lang):
        '''
        mode: str - The mode of the model (default or finetuned)
        lang: str - The language of the model (eng or kor) 
        '''
        self.mode = mode
        self.lang = lang
        
    def __call__(self):
        self.chpwd()
        return self.path(self.mode, self.lang)
    def chpwd(self):
        if not os.path.exists("README.md"):
            print("README.md 파일이 있는 디렉토리에서 실행해주세요.")
            exit()
    def path(self,mode,lang):
        if mode == "default":
            return "google/gemma-2-2b-it" # 기본적인 모델
        elif lang == "eng":
            return "pengsu/MLB-care-for-mind-eng"
        elif lang == "kor":
            return "pengsu/MLB-care-for-mind-kor"
    
