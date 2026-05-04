import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('data/Ice Cream Sales and Temperature.csv')
print(df)
df.drop(3,inplace=True)
df.drop(40,inplace=True)
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df['Hour'] = df['Datetime'].dt.hour
df['minute'] = df['Datetime'].dt.minute
df['total_min'] = df['Hour'] * 60 + df['minute']

x = df[['Temperature (Celsius)','total_min']]
y = df['Ice Cream Sales']

model = LinearRegression()
model.fit(x,y)
score = model.score(x, y)

predict_min = 8 * 60 + 15
predict_min_2 = 11 * 60 + 20 
pred = model.predict([[21,predict_min],[58,predict_min_2]])

print(f'Погрешность: {score}')
print(f'predict: {round(pred[0],2)} {round(pred[1],2)}')


poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)
model_2 = LinearRegression()
model_2.fit(x_poly,y)
score_poly = model_2.score(x_poly,y)

x_new = [[21, predict_min],
         [58, predict_min_2]]

x_new_poly = poly.transform(x_new)
pred_poly = model_2.predict(x_new_poly)

print(f'Погрешность: {score_poly}')
print(f'predict: {round(pred_poly[0],2)} {round(pred_poly[1],2)}')