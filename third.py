from transformers import AutoProcessor, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline
import torch

# Gemma 2 모델 경로 및 Stable Diffusion 모델 경로
gemma_model_path = "google/gemma-2-2b-it"
sd_model_path = "CompVis/stable-diffusion-v1-4"  # Stable Diffusion 모델 경로

# 로컬 경로에서 Gemma 2 모델과 프로세서 로드
processor = AutoProcessor.from_pretrained(gemma_model_path)
model = AutoModelForCausalLM.from_pretrained(
    gemma_model_path,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

# Stable Diffusion 모델 로드
sd_pipe = StableDiffusionPipeline.from_pretrained(sd_model_path)
sd_pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# 사용자로부터 텍스트 프롬프트 입력 받기
text_prompt = input("당신의 현재 감정을 한 문장으로 입력하세요: ")

# Gemma 2 모델을 사용한 감정 분석 수행
input_ids = processor(text_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**input_ids, max_new_tokens=32)
generated_text = processor.decode(outputs[0])

# 감정 분석 결과 확인 (부정적인 감정 또는 긍정적인 감정 추출)
if "sad" in generated_text.lower() or "angry" in generated_text.lower() or "depressed" in generated_text.lower() or "devasted" in generated_text.lower():
    emotion = "negative"
else:
    emotion = "positive"

# 감정에 따른 이미지 프롬프트 및 감동적인 문구 정의
if emotion == "negative":
    # 부정적인 감정일 때 기분을 좋게 하는 이미지 생성 (풍경 및 감동적인 명언 출력)
    uplifting_image_prompt = "A peaceful landscape, famous travel destinations, or a beautiful sunset over the ocean."
    inspirational_quote = "힘든 순간은 지나갈 것입니다. 당신은 이겨낼 힘을 가지고 있습니다."
    print(f"기분을 좋게 하는 이미지 프롬프트: {uplifting_image_prompt}")
    generated_image = sd_pipe(uplifting_image_prompt).images[0]
    print(f"감동적인 명언: {inspirational_quote}")
elif emotion == "positive":
    # 긍정적인 감정일 때 현재 감정을 나타내는 이미지 생성
    joyful_image_prompt = "stunning scenary picture"
    print(f"긍정적인 감정을 나타내는 이미지 프롬프트: {joyful_image_prompt}")
    generated_image = sd_pipe(joyful_image_prompt).images[0]

# 생성된 이미지를 저장
image_save_path = "emotion_based_image.png"
generated_image.save(image_save_path)

# 결과 확인 출력
print(f"감정 분석에 따른 이미지가 성공적으로 저장되었습니다: {image_save_path}")
