# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from diffusers import StableDiffusionPipeline
# import torch
# import os

# # GPU 0만 사용하도록 설정
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# # Stable Diffusion 모델 경로
# sd_model_path = "CompVis/stable-diffusion-v1-4"  # Stable Diffusion 기본 모델 경로

# # 감정 분석 모델 경로
# sentiment_model_path = "/home/mmai6k_02/workspace/personal_practice/gemma-2-2b-it"  # 감정 분석 모델 경로

# # 감정 분석 모델 및 토크나이저 로드
# tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
# sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)

# # Stable Diffusion 파이프라인 로드
# device = "cuda" if torch.cuda.is_available() else "cpu"
# sd_pipe = StableDiffusionPipeline.from_pretrained(sd_model_path)
# sd_pipe.to(device)

# # 사용자로부터 텍스트 프롬프트 입력 받기
# text_prompt = input("당신의 현재 감정을 한 문장으로 입력하세요: ")

# # 감정 분석 수행 함수
# def sentiment_model_analysis(text_prompt):
#     inputs = tokenizer(text_prompt, return_tensors="pt")  # 토크나이저로부터 입력 텐서를 생성
#     inputs = {key: value.to(device) for key, value in inputs.items()}  # 모든 입력 텐서를 GPU로 이동
#     sentiment_model.to(device)  # 감정 분석 모델을 GPU로 이동
#     outputs = sentiment_model(**inputs)  # 모델과 입력이 동일한 디바이스에 있어야 함
#     logits = outputs.logits
#     predicted_class = torch.argmax(logits, dim=1).item()

#     if predicted_class == 1:
#         return "positive"
#     else:
#         return "negative"

# # 감정 분석 수행
# emotion = sentiment_model_analysis(text_prompt)

# # 감정에 따른 이미지 생성 프롬프트 정의
# if emotion == "negative":
#     image_prompt = "A peaceful landscape, famous travel destinations, or a beautiful sunset over the ocean."
#     print(f"부정적인 감정 분석. 이미지 프롬프트: {image_prompt}")
# elif emotion == "positive":
#     image_prompt = "A stunning scenery picture with bright colors."
#     print(f"긍정적인 감정 분석. 이미지 프롬프트: {image_prompt}")

# # Stable Diffusion을 사용한 이미지 생성
# generated_image = sd_pipe(image_prompt, guidance_scale=7.5).images[0]

# # 생성된 이미지를 저장
# image_save_path = "sd_generated_image.png"
# generated_image.save(image_save_path)

# # 결과 확인 출력
# print(f"Stable Diffusion 모델을 사용하여 생성된 이미지가 성공적으로 저장되었습니다: {image_save_path}")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from diffusers import StableDiffusionPipeline
import torch
import os

# GPU 0만 사용하도록 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Stable Diffusion 모델 경로
sd_model_path = "CompVis/stable-diffusion-v1-4"  # Stable Diffusion 기본 모델 경로

# 감정 분석 모델 경로 (감정 분석에 더 적합한 모델 사용) bert model 사용 
sentiment_model_path = "nlptown/bert-base-multilingual-uncased-sentiment"

# 감정 분석 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)

# Stable Diffusion 파이프라인 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
sd_pipe = StableDiffusionPipeline.from_pretrained(sd_model_path)
sd_pipe.to(device)

# 사용자로부터 텍스트 프롬프트 입력 받기
text_prompt = input("당신의 현재 감정을 한 문장으로 입력하세요: ")

# 감정 분석 수행 함수
def sentiment_model_analysis(text_prompt):
    inputs = tokenizer(text_prompt, return_tensors="pt")  # 토크나이저로부터 입력 텐서를 생성
    inputs = {key: value.to(device) for key, value in inputs.items()}  # 모든 입력 텐서를 GPU로 이동
    sentiment_model.to(device)  # 감정 분석 모델을 GPU로 이동
    outputs = sentiment_model(**inputs)  # 모델과 입력이 동일한 디바이스에 있어야 함
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # 모델이 0~4 사이의 값을 반환하는 경우에 따라 감정 결정
    if predicted_class >= 3:
        return "positive"
    else:
        return "negative"

# 감정 분석 수행
emotion = sentiment_model_analysis(text_prompt)

# 감정에 따른 이미지 생성 프롬프트 정의
if emotion == "negative":
    image_prompt = "A peaceful landscape, famous travel destinations, or a beautiful sunset over the ocean."
    print(f"부정적인 감정 분석. 이미지 프롬프트: {image_prompt}")
elif emotion == "positive":
    image_prompt = "A stunning scenery picture with bright colors."
    print(f"긍정적인 감정 분석. 이미지 프롬프트: {image_prompt}")

# Stable Diffusion을 사용한 이미지 생성
generated_image = sd_pipe(image_prompt, guidance_scale=7.5).images[0]

# 생성된 이미지를 저장
image_save_path = "sd_generated_image.png"
generated_image.save(image_save_path)

# 결과 확인 출력
print(f"Stable Diffusion 모델을 사용하여 생성된 이미지가 성공적으로 저장되었습니다: {image_save_path}")
