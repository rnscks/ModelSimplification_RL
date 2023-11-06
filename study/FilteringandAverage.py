import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 예제 데이터 생성
data = {
    'Year': [2010, 2011, 2012, 2013, 2014, 2012],
    'Value': [10, 15, 12, 18, 20, 23],
    'k': ["1", "1", "2", "1", "2", "2"],
    'x': ['A', 'B', 'A', 'B', 'A', 'B']
}

df = pd.DataFrame(data)

# 데이터프레임을 그룹화하고 각 Year에 대한 평균 계산
grouped = df.groupby(['Year', 'k', 'x'])['Value'].mean().reset_index()

# Seaborn을 사용하여 선 그래프 그리기
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))

# 선 그래프 그리기
sns.lineplot(data=grouped, x='k', y='Value', hue='x', palette='Set1')

plt.title('Line Plot with Average for each Year')
plt.xlabel('K')
plt.ylabel('Value')
plt.legend(title='x')
plt.show()
