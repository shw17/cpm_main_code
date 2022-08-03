import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')


# feature encoding
# RENDER_NBUMBER, CONNECTION_DOWNLINK
def percentile_encoding(df, column_name):
    bins_height = [df[column_name].min(), df[column_name].quantile(.25), df[column_name].quantile(.50),
                   df[column_name].quantile(.75), df[column_name].max()]
    category = ['lower', 'low', 'medium', 'high']
    # convert numerical to category
    df[column_name] = pd.cut(df[column_name], bins_height, labels=category, include_lowest=True)
    # label encoding
    df[f'{column_name}_ENCODE'] = df[column_name].astype('category').cat.codes
    return df, dict(enumerate(df[column_name].astype('category').cat.categories))


# AD_TYPE, DAY
def one_hot_encoding(df, column_name):
    df = pd.get_dummies(df, columns=column_name, drop_first=True)
    return df


# COUNTRY_GROUP, BROWSER_GROUP, DEVICE_TYPE
def frequency_encoding(df, column_name):
    fe = df.groupby(column_name).size()
    co_ = fe / df.shape[0]
    df[f'{column_name}_FE'] = df[column_name].map(co_).round(2)
    return df, dict(co_.round(2))


# UNIT, UNIT_SUBTYPE
def label_encoding(df, column_name):
    df[f'{column_name}_ENCODE'] = df[column_name].astype('category').cat.codes()
    return df, dict(enumerate(df[f'{column_name}'].astype('category').cat.categories))


# drop old columns and sqrt CPM
def drop_function(df, column_name):
    ppp = df.pop('CPM')
    df.insert(df.shape[1], 'CPM', ppp)
    df['CPM'] = df['CPM'] ** 0.5
    df = df.drop(column_name, axis=1)
    return df


# check over-fitting
deepth = [i for i in range(1,10)]
r2 = []
rmsetest = []
r2test = []
for i in deepth:
    forest = GradientBoostingRegressor(max_depth=i)
    _ = forest.fit(X_train, y_train)
    r2_train = forest.score(X_train, y_train)
    r2_test= forest.score(X_test, y_test)
    pred = forest.predict(X_test)
    rmse = mean_squared_error(y_test, pred)
    r2.append(r2_train)
    r2test.append(r2_test)
    rmsetest.append(rmse)

pyplot.figure(1)
pyplot.plot(deepth, r2, '-o', label='Train')
pyplot.plot(deepth, r2test, '-o', label='Test')
pyplot.plot(deepth, rmsetest, '-*', label='mse')
pyplot.title('gb')
pyplot.legend()
pyplot.show()


# save and load models
forest = GradientBoostingRegressor(max_depth=i)
_ = forest.fit(X_train, y_train)
pickle.dump(forest, open(f"{modelname}.pickle.dat", "wb"))
loaded_model = pickle.load(open(f"{modelname}.pickle.dat", "rb"))
pred = loaded_model.predict(X_test)


# reference table
d_table = df.iloc[:, :-1].loc[:, ['HOUR', 'RENDERENCODE', 'ADTYPE_display', 'COUNTRYFE',
                                  'BROWSERFE', 'DEVICEFE', 'UNITENCODE']]
d_table.insert(3, 'DAY', df.DAY)
index = pd.MultiIndex.from_product([d_table[f'{col}'].unique() for col in d_table.columns],
                                   names=d_table.columns.tolist())
table = pd.DataFrame(index=index).reset_index()


# map numerical features back to categorical in the reference table
# the main idea is to use dictionary
table['COUNTRY_GROUP'] = table['COUNTRYFE'].map(dict([(value, key) for key, value in frequency_encoding(df, 'COUNTRY_GROUP')[1].items()]))
table['BROWSER_GROUP'] = table['BROWSERFE'].map(dict([(value, key) for key, value in frequency_encoding(df, 'BROWSER_GROUP')[1].items()]))
table['DEVICE_GROUP'] = table['DEVICEFE'].map(dict([(value, key) for key, value in frequency_encoding(df, 'DEVICE_GROUP')[1].items()]))
table['UNIT'] = table['UNITENCODE'].map(label_encoding(df, 'UNIT')[1])
table['AD_TYPE'] = table['ADTYPE_display'].map({1: 'display'})
table['RENDER_NUMBER'] = table['RENDERENCODE'].map(percentile_encoding(df, 'RENDER_NUMBER'))
column_filter = ['HOUR', 'DAY', 'COUNTRY_GROUP', 'BROWSER_GROUP', 'DEVICE_GROUP', 'UNIT', 'AD_TYPE', 'RENDER_NUMBER']
table_final = table.loc[:, column_filter]
table_final['CPM_PREDICT'] = pred



