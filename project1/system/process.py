import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from diffusers import StableDiffusionPipeline

class EmotionImageProcessor:
    def __init__(self,path):
        # 모델 경로 설정
        self.sd_model_path = "CompVis/stable-diffusion-v1-4"  # Stable Diffusion 기본 모델 경로
        self.sentiment_model_path = path
        self.lang = path.split("/")[-1].split("_")[1]

        # GPU 설정
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 감정 분석 모델 및 토크나이저 로드
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.sentiment_model_path)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(self.sentiment_model_path)
            self.sentiment_model.to(self.device)
        except Exception as e:
            print("모델을 불러오는 중 오류가 발생했습니다. 저장된 파인튜닝 모델을 확인해주세요.")

        # Stable Diffusion 파이프라인 로드
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(self.sd_model_path)
        self.sd_pipe.to(self.device)

    # 감정 분석 수행 함수
    def analyze_sentiment(self, text_prompt):
        inputs = self.tokenizer(text_prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.sentiment_model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        output = self.is_negative(predicted_class)
        self.sentiment_model.to("cpu")
        self.delmodel("sentiment")
        print( "Predicted class:", output)
        return output

    # 감정에 따라 이미지 프롬프트 결정 함수
    def get_image_prompt(self, emotion):
        if emotion == "negative":
            return "A peaceful landscape, famous travel destinations, or a beautiful sunset over the ocean."
        else:
            return "A stunning scenery picture with bright colors."

    # 이미지 생성 및 반환 함수
    def generate_image(self, prompt, guidance_scale=7.5):
        image = self.sd_pipe(prompt, guidance_scale=guidance_scale).images[0]
        self.sd_pipe.to("cpu")
        self.delmodel("diffusion")
        return image
    
    def delmodel(self,mode):
        if mode == "sentiment":
            del self.sentiment_model
        elif mode == "diffusion":
            del self.sd_pipe

    def is_negative(self, predicted_class):
        if self.lang == "kor":
            return "negative" if predicted_class in [1, 3] else "positive"
        else:
            return "negative" if predicted_class == 1 else "positive"
