import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


df = pd.read_csv('data/salaries.csv')

print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df.describe())
df_2 = df.drop_duplicates()



# # базовая аналитика
print('\n\n------------ Базовая аналитика ------------ \n\n')
print(f"ЗП в разные годы {df.groupby('work_year')['salary_in_usd'].mean().round(2)}")
print(f"Средняя ЗП: {round(df['salary_in_usd'].mean(),2)}")
print(f'Медиана ЗП: {round(df["salary_in_usd"].median(),2 )}')
print(f"Распределение по уровням опыта:\n{df['experience_level'].value_counts()}\n")
print(f"Распределение по специальностям:\n{df['job_title'].value_counts()}\n")

print(f"Средняя ЗП по уровню и профессиям:\n{df.groupby(['experience_level','job_title'])['salary_in_usd'].mean().round(2).sort_values()}\n")
print(f"Разброс зарплат по ролям:\n{df.groupby(['experience_level','job_title'])['salary_in_usd'].agg(['min', 'max', 'std']).round(2)}")

print(f"Средняя ЗП по странам:\n{df.groupby('company_location')[['salary','salary_in_usd']].mean().round(2)}")
print(f"Топ стран с самыми высокими ЗП:\n{df.groupby('company_location')[['salary','salary_in_usd']].mean().round(2).sort_values(by='salary_in_usd', ascending=False).head()}")

print(f"Зарплата по размеру компании:\n{df_2.groupby('company_size')['salary_in_usd'].agg(['median', 'mean', 'count']).round(2).sort_values(by='median', ascending=False)}\n")
print(f"Зарплата vs Формат работы: \n{df_2.groupby('remote_ratio')['salary_in_usd'].agg(['median', 'mean', 'count']).round(2).sort_values(by='median', ascending=False)}\n")

print('\n\n\n')

# #Графики
# print("\n\n--------Графики----------\n\n")

year_mean = df.groupby('work_year')['salary_in_usd'].mean().round(2)
sns.set_theme(style="darkgrid") 
plt.figure(figsize=(10, 5))
ax = year_mean.plot(
    kind='line', 
    marker='o', 
    linewidth=3, 
    markersize=8, 
    color='#FF7F0E',
    title='ЗП в разные годы'
)
ax.set_xlabel('Год')
ax.set_ylabel('Средняя ЗП (USD)')
plt.show()


sns.set_theme(style="darkgrid") 

top_5_jobs = df_2['job_title'].value_counts().head(5).index

df_top = df_2[df_2['job_title'].isin(top_5_jobs)]

pivot_df = df_top.pivot_table(
    index='job_title', 
    columns='experience_level', 
    values='salary_in_usd', 
    aggfunc='mean'
)

pivot_df = pivot_df[['EN', 'MI', 'SE', 'EX']]

ax = pivot_df.plot(
    kind='bar', 
    figsize=(12, 6), 
    colormap='viridis',
    width=0.8
)

ax.set_title('Средняя ЗП по грейдам (Топ-5 профессий)', fontsize=14)
ax.set_xlabel('Профессия')
ax.set_ylabel('Средняя ЗП (USD)')
plt.xticks(rotation=0)
plt.show()

print('\n\n----------Предикт уровня заработной платы--------------\n\n')

df_2['exp_num'] = df_2['experience_level'].map({'EN': 1, 'MI': 2, 'SE': 3, 'EX': 4})
df_2['size_num'] = df_2['company_size'].map({'S': 1, 'M': 2, 'L': 3})

x = df_2[['work_year', 'exp_num', 'remote_ratio', 'size_num']]
y = df_2['salary_in_usd']

fist_model = LinearRegression()
fist_model.fit(x.values,y)

#Точность 
score = fist_model.score(x, y)
print(f"Точность: {score}")

pred = fist_model.predict([[2026, 3, 100, 2]])
#[2026, 3, 100, 2] 2026 году, уровень - senior, 100 - полная удаленка, 2 - среднего размера бизнем \ компания 
 
print(round(pred[0],2))