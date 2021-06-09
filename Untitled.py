#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
past = 14 #дней в прошлое
future = 7 #дней в будущее
test_lines = 20 #количество строк для теста


# In[5]:


rates = pandas.read_excel('Desktop/us.xlsx')


# In[6]:


rates.curs.plot()


# In[7]:


dollar_rate = rates.curs


# In[8]:


start = past
end = len(dollar_rate) - future
count = len(dollar_rate)
print(f"Всего {count} значений, для обучения используем: с {start} по {end}")


# In[9]:


tab = []
for i in range(start, end):
    s = dollar_rate[i-past: i+future]
    tab.append(list(s))


# In[10]:


past_columns = []
for i in range(past):
    past_columns.append(f"past_{i}")


# In[11]:


past_columns


# In[12]:


future_columns = []
for i in range(future):
    future_columns.append(f"future_{i}")


# In[13]:


future_columns


# In[14]:


data_frame = pandas.DataFrame(data = tab, columns = (past_columns + future_columns)) 


# In[15]:


X = data_frame[past_columns][:-test_lines]   #данные для предсказания
Y = data_frame[future_columns][:-test_lines] #данные что мы предсказываем


# In[16]:


X_test = data_frame[past_columns][-test_lines:] 
Y_test = data_frame[future_columns][-test_lines:]


# In[17]:


import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


# In[18]:


new_df = pandas.DataFrame(columns = ['model_name', 'average_err', 'min_err', 'max_err'])


# In[19]:


def fit_model (model, model_name):
    model.fit(X, Y)
    sum_err = 0
    min_err = 100
    max_err = 0
    for i in range(test_lines):
        prediction = model.predict([X_test.iloc[i]])
        err = mean_absolute_error(prediction[0], Y_test.iloc[i])
        print(f"date #{i}, error = {err}")
        plt.plot(prediction[0], label = "Prediction")
        plt.plot(Y_test.iloc[i], label = "Real data")
        plt.legend()
        plt.show()
        sum_err += err
        max_err = max([max_err, err])
        min_err = min([min_err, err])
    average_err = sum_err / test_lines
    new_df.loc[len(new_df)] = [model_name, average_err, min_err, max_err]
    return average_err #средняя ошибка


# In[20]:


from sklearn.linear_model import LinearRegression


# In[21]:


LR = LinearRegression() 
LR_average_err = fit_model(LR, 'LR')


# In[22]:


print(f'Средняя ошибка метода составила: {LR_average_err}')


# In[23]:


from sklearn.neighbors import KNeighborsRegressor #Метод ближайших соседей
KNN = KNeighborsRegressor(16) #Установили количество ближайших соседей
KNN_average_err = fit_model(KNN, 'KNN')


# In[24]:


print(f'Средняя ошибка метода составила: {KNN_average_err}')


# In[25]:


from sklearn.neural_network import MLPRegressor #Нейронная сеть
#MLPR = MLPRegressor(hidden_layer_sizes=(256,128,64,32), max_iter=10000, random_state = 42)
MLPR = MLPRegressor(hidden_layer_sizes=(119,127,71,63,32), max_iter=1000, random_state = 42) 
MLPR_average_err = fit_model(MLPR, 'MLPR')


# In[26]:


print(f'Средняя ошибка метода составила: {MLPR_average_err}')


# In[27]:


from sklearn.dummy import DummyRegressor    #Простые правила
Dummy = DummyRegressor (strategy="median") #Сменили стратегию
Dummy_average_err = fit_model(Dummy, 'Dummy')


# In[28]:


print(f'Средняя ошибка метода составила: {Dummy_average_err}')


# In[29]:


from sklearn.cross_decomposition import PLSRegression #Перекрестное разложение
PLSR = PLSRegression (n_components=5, max_iter = 1000) #Изменили количество компонентов и итераций
PLSR_average_err = fit_model(PLSR, 'PLSR')


# In[30]:


print(f'Средняя ошибка метода составила: {PLSR_average_err}')


# In[36]:


from sklearn.cross_decomposition import CCA #кореляционный анализ
CCA = CCA(n_components = 2, max_iter = 2000)
CCA_average_err = fit_model(CCA, 'CCA')


# In[37]:


print(f'Средняя ошибка метода составила: {CCA_average_err}')


# In[38]:


for i in range(len(new_df)):
    print(f'Модель {new_df.model_name[i]} показала результат: {new_df.average_err[i]} средней ошибки.')
    print(f'Минимальная ошибка составила: {new_df.min_err[i]}, максимальная ошибка составила: {new_df.max_err[i]}')
    print('')


# In[39]:


new_df.set_index('model_name').plot(kind='bar')


# In[40]:


new_df.set_index('model_name').plot.barh(stacked=True)


# In[ ]:





# In[ ]:




