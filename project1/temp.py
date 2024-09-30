import torch

class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob = 0.5  # 드롭아웃 확률

        # L1: 첫 번째 합성곱층 (Conv Layer)
        # 입력 이미지 형태: (?, 28, 28, 1)
        # Conv2d: 출력 채널 32개, 커널 크기 3x3, 스트라이드 1, 패딩 1
        # ReLU: 활성화 함수
        # MaxPool2d: 커널 크기 2x2, 스트라이드 2로 다운샘플링 -> 출력 형태: (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # L2: 두 번째 합성곱층 (Conv Layer)
        # 입력 이미지 형태: (?, 14, 14, 32)
        # Conv2d: 출력 채널 64개, 커널 크기 3x3, 스트라이드 1, 패딩 1
        # ReLU: 활성화 함수
        # MaxPool2d: 커널 크기 2x2, 스트라이드 2로 다운샘플링 -> 출력 형태: (?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # L3: 세 번째 합성곱층 (Conv Layer)
        # 입력 이미지 형태: (?, 7, 7, 64)
        # Conv2d: 출력 채널 128개, 커널 크기 3x3, 스트라이드 1, 패딩 1
        # ReLU: 활성화 함수
        # MaxPool2d: 커널 크기 2x2, 스트라이드 2, 패딩 1로 다운샘플링 -> 출력 형태: (?, 4, 4, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        # L4: 첫 번째 선형층 (Fully Connected Layer)
        # 입력 노드 수: 4x4x128, 출력 노드 수: 625
        # ReLU: 활성화 함수
        # Dropout: 드롭아웃으로 과적합 방지, p=0.5
        self.fc1 = torch.nn.Linear(4 * 4 * 128, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)  # 가중치 초기화
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - self.keep_prob))

        # L5: 최종 선형층 (Fully Connected Layer)
        # 입력 노드 수: 625, 출력 노드 수: 10 (클래스 개수)
        self.fc2 = torch.nn.Linear(625, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)  # 가중치 초기화

    def forward(self, x):
        out = self.layer1(x)  # 첫 번째 합성곱층 통과
        out = self.layer2(out)  # 두 번째 합성곱층 통과
        out = self.layer3(out)  # 세 번째 합성곱층 통과
        out = out.view(out.size(0), -1)  # 선형층에 입력하기 위해 텐서를 Flatten
        out = self.layer4(out)  # 첫 번째 선형층 통과
        out = self.fc2(out)  # 최종 선형층 통과
        return out  # 최종 출력 반환

model = CNN().to('cuda')
print(model)