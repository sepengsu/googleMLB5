from transformers import TrainerCallback
import matplotlib.pyplot as plt
import os
class LossHistoryCallback(TrainerCallback):
    def __init__(self):
        self.loss_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        # 로그에 'loss'가 있을 때만 저장
        if 'loss' in logs:
            self.loss_history.append(logs['loss'])


# 그래프를 output_dir에 저장하는 함수
def save_loss_plot(loss_history, output_dir):
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    
    # output_dir 경로가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 그래프를 output_dir에 저장
    output_file = os.path.join(output_dir, "training_loss.png")
    plt.savefig(output_file)
    print(f"손실 그래프가 {output_file} 경로에 저장되었습니다.")
    plt.close()