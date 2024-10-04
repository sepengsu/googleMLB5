import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from diffusers import StableDiffusionPipeline

class EmotionImageProcessor:
    def __init__(self,path,lang):
        # 모델 경로 설정
        self.sd_model_path = "CompVis/stable-diffusion-v1-4"  # Stable Diffusion 기본 모델 경로
        self.sentiment_model_path = path
        self.lang = lang

        # GPU 설정
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 감정 분석 모델 및 토크나이저 로드
        try:
            lend = 2 if self.lang == "eng" else 4
            self.tokenizer = AutoTokenizer.from_pretrained(self.sentiment_model_path)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                self.sentiment_model_path, num_labels=lend,)
            self.sentiment_model.to(self.device)
        except Exception as e:
            print("모델을 불러오는 중 오류가 발생했습니다. 저장된 파인튜닝 모델을 확인해주세요.")
        else:
            print(f"모델을 {self.sentiment_model_path}에서 성공적으로 불러왔습니다.")

        # Stable Diffusion 파이프라인 로드
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(self.sd_model_path)
        self.sd_pipe.to(self.device)

    # 감정 분석 수행 함수
    def analyze_sentiment(self, text_prompt):
        self.sentiment_model.to(self.device)
        inputs = self.tokenizer(text_prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.sentiment_model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
        probabilities = torch.softmax(logits, dim=-1).tolist()[0]
        output = self.is_negative(predicted_class)
        inputs = None
        self.sentiment_model.to("cpu")
        self.sentiment_model = None
        print( "Predicted class:", output, "Probabilities:", probabilities)
        return output

    # 감정에 따라 이미지 프롬프트 결정 함수
    def get_image_prompt(self, emotion):
        if emotion == "negative":
            return "A peaceful landscape, famous travel destinations, or a beautiful sunset over the ocean."
        else:
            return "A stunning scenery picture with bright colors."

    # 이미지 생성 및 반환 함수
    def generate_image(self, prompt, guidance_scale=7.5):
        self.sd_pipe.to(self.device)
        image = self.sd_pipe(prompt, guidance_scale=guidance_scale).images[0]
        self.sd_pipe= None
        return image

    def is_negative(self, predicted_class):
        if self.lang == "kor":
            return "negative" if predicted_class in [1, 3] else "positive"
        else:
            return "negative" if predicted_class == 0 else "positive"
