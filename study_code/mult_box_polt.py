import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random

data = {
    'k': [],
    'Experiment': [],
    'Result': [],
    'Group1': []
}

num_models = 14
num_experiments_per_model = 14

for model in range(1, num_models + 1):

    for experiment in range(1, num_experiments_per_model + 1):
        data['k'].append(model)    
        result = random.uniform(0, 1)
        data['Experiment'].append(experiment)
        data['Result'].append(result)
        data['Group1'].append('group1')

for model in range(1, num_models + 1):
    for experiment in range(1, num_experiments_per_model + 1):
        data['k'].append(model)    
        result = random.uniform(0, 1)
        data['Experiment'].append(experiment)
        data['Result'].append(result)
        data['Group1'].append('group2')     

for model in range(1, num_models + 1):
    for experiment in range(1, num_experiments_per_model + 1):
        data['k'].append(model)    
        result = random.uniform(0, 1)
        data['Experiment'].append(experiment)
        data['Result'].append(result)
        data['Group1'].append('group3') 

df = pd.DataFrame(data)
print(df.head())
# 박스 플롯 그리기
sns.set(style="whitegrid")
plt.figure(figsize=(40, 30))

# 첫 번째 서브플롯
plt.subplot(2, 2, 1)
sns.boxplot(data=df, x='k', y='Result', hue='Group1', palette="Set1").tick_params(axis='both',labelsize=30)
plt.title('Box Plot 1', fontsize = 45)
plt.xlabel('K', fontsize = 40)
plt.ylabel('Result', fontsize = 40)
plt.legend(fontsize=30,loc='upper right')

# 두 번째 서브플롯
plt.subplot(2, 2, 2)
sns.boxplot(data=df, x='k', y='Result', hue='Group1', palette="Set2").tick_params(axis='both', labelsize=30)
plt.title('Box Plot 2', fontsize = 45)
plt.xlabel('K', fontsize = 40)
plt.ylabel('Result', fontsize = 40)

# 세 번째 서브플롯
plt.subplot(2, 2, 3)
sns.boxplot(data=df, x='k', y='Result', hue='Group1', palette="Set2").tick_params(axis='both', labelsize=30)
plt.title('Box Plot 2', fontsize = 45)
plt.xlabel('K', fontsize = 40)
plt.ylabel('Result', fontsize = 40)

# 네 번째 서브플롯
plt.subplot(2, 2, 4)
sns.boxplot(data=df, x='k', y='Result', hue='Group1', palette="Set2").tick_params(axis='both', labelsize=30)
plt.title('Box Plot 2', fontsize = 45)
plt.xlabel('K', fontsize = 40)
plt.ylabel('Result',fontsize = 40)



plt.tight_layout()
plt.savefig("fig1")
