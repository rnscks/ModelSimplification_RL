import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random

data = {
    'k': [],
    'Experiment': [],
    'Result': []
}

# 5개의 모델과 각 모델당 10개의 실험 결과 생성
num_models = 5
num_experiments_per_model = 10

for model in range(1, num_models + 1):
    for experiment in range(1, num_experiments_per_model + 1):
        data['k'].append(model)
        result = random.uniform(0, 1)  # 무작위 실험 결과 (실제 데이터로 대체)
        data['Experiment'].append(experiment)
        data['Result'].append(result)

df = pd.DataFrame(data)
grouped_df = df.groupby('k')
print(grouped_df)

# 박스 플롯 그리기
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))
sns.boxplot(data=data, x='k', y='Result')
plt.title('Box Plot of Experiment Results by Model')
plt.xlabel('K')
plt.ylabel('Result')

plt.show()
