import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 읽기 (파일 경로를 수정하세요)
file_path = "/path/to/your/file.csv"

# CSV 데이터 로드
df = pd.read_csv(file_path)

# 데이터 확인 (특정 열이 "Step"과 "Value"인지 확인 필요)
print(df.head())

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(df['Step'], df['Value'], marker='o', label='Accuracy')

# 축과 제목 설정
plt.xlabel("Epoch (Step)")
plt.ylabel("Accuracy (Value)")
plt.title("Accuracy vs Epoch")
plt.legend()
plt.grid(True)

# 그래프 표시
plt.show()