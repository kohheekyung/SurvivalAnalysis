from lifelines import KaplanMeierFitter
import pandas as pd
import numpy as np


data = pd.read_csv('./data/2020010629_Imputed.csv')
data_X = data['RID'] #data.drop(['CONV_TIME'], axis=1)
data_y = data['CONV_TIME']



print('Converting', 'CONV_TIME', 'to Multi-hot encoding')

#get 1-hot and replace original column with the >= 2 categories as columns
one_hot_df = pd.get_dummies(data_y)
print(one_hot_df.columns)


for i in range(len(one_hot_df.columns)):
    label = one_hot_df.columns[i]
    data_y = data_y.apply(lambda x: i + 1  if x==label else x)
    print(label, i)

time = np.array(data_y * 6)
print(time)
event = np.ones(data['RID'].shape[0])

print(time.shape)
print(event.shape)

kmf = KaplanMeierFitter()
_ =kmf.fit(time, event)

plot = kmf.plot_survival_function()
plot.set_xlabel('Month')
plot.set_ylabel('S(t | x)')
#plot.savefig('./surv_func_KM')