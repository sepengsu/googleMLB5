import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
from system.process import EmotionImageProcessor
from system.reader import ModelReader
class EmotionImageApp:
    def __init__(self, window):
        # Tkinter 창 설정
        self.window = window
        self.window.title("감정 분석 및 이미지 생성")
        self.window.geometry("500x700")

        # 출력 라벨
        self.output_label = tk.Label(window, text="감정을 입력하세요", font=("Helvetica", 16))
        self.output_label.pack(pady=20)

        # 언어 모드 선택 버튼 (수평 정렬)
        self.language_mode = tk.StringVar(value="eng")
        language_frame = tk.Frame(window)  # 수평 정렬을 위한 프레임
        self.language_button_eng = tk.Radiobutton(language_frame, text="English", variable=self.language_mode, value="eng", font=("Helvetica", 12))
        self.language_button_kor = tk.Radiobutton(language_frame, text="Korean", variable=self.language_mode, value="kor", font=("Helvetica", 12))
        self.language_button_eng.pack(side="left", padx=5)
        self.language_button_kor.pack(side="left", padx=5)
        language_frame.pack(pady=10)

        # 모델 선택 버튼 (수평 정렬)
        self.model_mode = tk.StringVar(value="default")
        model_frame = tk.Frame(window)  # 수평 정렬을 위한 프레임
        self.model_button_default = tk.Radiobutton(model_frame, text="Default Model", variable=self.model_mode, value="default", font=("Helvetica", 12))
        self.model_button_finetuned = tk.Radiobutton(model_frame, text="Finetuned Model", variable=self.model_mode, value="finetuned", font=("Helvetica", 12))
        self.model_button_default.pack(side="left", padx=5)
        self.model_button_finetuned.pack(side="left", padx=5)
        model_frame.pack(pady=10)
        
        # 상태 입력 필드
        self.entry = tk.Entry(window, font=("Helvetica", 14), width=40)
        self.entry.pack(pady=10)

        # 입력 버튼
        self.submit_button = tk.Button(window, text="입력", command=self.show_status, font=("Helvetica", 12))
        self.submit_button.pack(pady=10)

        # 이미지 또는 로딩 중 메시지를 표시할 라벨
        self.image_label = tk.Label(window)
        self.image_label.pack(pady=20)

        # 다운로드 버튼 (처음엔 숨김)
        self.download_button = tk.Button(window, text="이미지 다운로드", command=self.download_image, font=("Helvetica", 12))
        self.download_button.pack(pady=10)
        self.download_button.pack_forget()  # 처음에는 버튼 숨김

        # 엔터키로 입력을 처리
        window.bind('<Return>', self.show_status)

        # 감정 분석 및 이미지 생성 클래스 인스턴스
        model_type = self.model_mode.get()
        language = self.language_mode.get()
        path = ModelReader(model_type, language)()
        self.processor = EmotionImageProcessor(path)

        # 생성된 이미지를 저장하기 위한 변수
        self.generated_image = None

    # 이미지 Tkinter로 표시하기 위한 함수
    def display_image(self, img):
        img = img.resize((300, 300), Image.Resampling.LANCZOS)  # 이미지 크기 조정
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk  # 가비지 컬렉션 방지

    # 이미지 다운로드 처리 함수
    def download_image(self):
        if self.generated_image is not None:
            # 파일 저장 대화 상자를 통해 파일 저장 경로 선택
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG 파일", "*.png")])
            if file_path:
                self.generated_image.save(file_path)
                self.output_label.config(text="이미지가 저장되었습니다.")

    # 함수 실행 및 결과 출력
    def show_status(self, event=None):
        # 입력된 텍스트를 가져옴
        status = self.entry.get()

        # 이미지가 있으면 제거하고 "로딩 중" 메시지 표시
        self.output_label.config(text="로딩 중... 이미지 생성 중입니다.")
        self.image_label.config(image='')  # 기존 이미지 제거
        self.download_button.pack_forget()  # 다운로드 버튼 숨김

        # Tkinter의 after() 메서드를 사용해 이미지 생성 후에 처리
        self.window.after(100, lambda: self.process_and_display_image(status))

    # 감정 분석 및 이미지 생성 처리 후 결과를 표시하는 함수
    def process_and_display_image(self, status):
        # 감정 분석 및 이미지 생성 처리
        emotion = self.processor.analyze_sentiment(status)
        image_prompt = self.processor.get_image_prompt(emotion)
        self.generated_image = self.processor.generate_image(image_prompt)

        # 생성된 이미지를 Tkinter 창에 표시
        self.display_image(self.generated_image)

        # 감정 분석 결과를 출력 라벨에 표시
        if emotion == "negative":
            self.output_label.config(text="오늘 기분이 좋지 않으시군요")
        else:
            self.output_label.config(text="오늘 기분이 좋으시군요")
        # 다운로드 버튼 표시
        self.download_button.pack()

# Tkinter 창 생성 및 실행
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionImageApp(root)
    root.mainloop()
