import time
import numpy as np
import pandas as pd
from scipy.stats import stats
from sklearn.base import TransformerMixin
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, Imputer, MinMaxScaler
import pandas_profiling
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
import category_encoders as ce
from sklearn.svm import SVR
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# TODO: Handle outliers
# TODO: Handle high cardinality columns
# TODO: Feature selection library
# TODO: Hyperparameters optimise


class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')


class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


def populate_missing_data(dataset):
    # d = DataFrameImputer().fit_transform(dataset)
    d = dataset.copy()
    d['Age'].fillna(d['Age'].mean(),inplace=True)
    d['Profession'].fillna(method='bfill',inplace=True)
    d['Year of Record'].fillna(method='bfill',inplace=True)
    # d['Gender'].fillna(method='bfill',inplace=True)
    # d['Hair Color'].fillna(method='bfill',inplace=True)
    d['Gender'].fillna('Unknown',inplace=True)
    d['Hair Color'].fillna('Unknown',inplace=True)
    d['University Degree'].fillna('Unknown',inplace=True)
    # d.drop(axis=1,labels=['Gender','Hair Color'],inplace=True)
    return d


# handle negative price and outliers
def remove_rows(dataset):
    d = dataset.copy()
    # d.dropna(axis=0, subset=['Profession', 'Year of Record'], inplace=True)
    # d = d[d['Income in EUR'] > 0]
    # profCount = d['Profession'].value_counts()
    # counts = d['Country'].value_counts()
    # d = d[~d['Profession'].isin(profCount[profCount < 5].index) & ~d['Country'].isin(counts[counts < 20].index)]
    return d


"""
make gender consistent
handle uni degree 
TODO:: handle negative salaries
hair color
country
profession
"""
def make_data_consistent(dataset):
    d = dataset.copy()
    d['Gender'] = d['Gender'].replace('0', np.nan).replace('unknown',np.nan)
    d['University Degree'] = d['University Degree'].replace('0',np.nan)
    d['Hair Color'] = d['Hair Color'].replace('0',np.nan).replace('Unknown',np.nan)
    # d['Income in EUR'] = d['Income in EUR'].replace(np.nan,0)
    return d


def encode_categories(dataset,testDataset):
    train = dataset.copy()
    test = testDataset.copy()
    # d = pd.get_dummies(d, columns=catColumns, prefix = catColumns)
    # catColumns = ['Gender','University Degree','Profession','Country','Hair Color','Year of Record']
    # targetCatColumns = ['Gender','University Degree','Profession','Country','Hair Color','Year of Record']
    targetCatColumns = ['Profession', 'Country']
    ce_targetEncoder = ce.TargetEncoder()
    train[targetCatColumns] = ce_targetEncoder.fit_transform(train[targetCatColumns], train['Income in EUR'])
    test[targetCatColumns] = ce_targetEncoder.transform(test[targetCatColumns])

    catColumns = ['Gender','University Degree','Hair Color']
    train['train'] = 1
    test['train'] = 0
    combined = pd.concat([train,test],sort=False)
    df = pd.get_dummies(combined,columns=catColumns,prefix=catColumns)
    train = df[combined['train'] == 1]
    test = df[combined['train'] == 0]
    train.drop(['train'],axis=1,inplace=True)
    test.drop(['train','Income in EUR'],axis=1,inplace=True)

    return train,test


def scale_data(X):
    stScaler_ds = StandardScaler()
    X = stScaler_ds.fit_transform(X)
    return X


def split_data(X, y):
    # split data into training and test parts
    # ... for now, we use all of the data for training and testing
    testSize = 0.20
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=testSize)
    return (Xtrain, Xtest, ytrain, ytest)


def plot_corr_matrix(dataset):
    df = dataset.copy(deep=True)
    # df = df[df.columns.drop(list(df.filter(regex='Profession')))]
    corr = df.corr()
    corr.style.background_gradient(cmap='coolwarm').set_precision(2)
    fig, ax = plt.subplots(figsize=(df.columns.size, df.columns.size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.savefig('plot.png')
    plt.show()


def predict_data(model, dataset, featureCol, outputFile):
    print("Predicting values")
    cols = featureCol.copy()
    instanceIds = dataset['Instance'].values
    data = dataset.loc[:, dataset.columns.str.contains('|'.join(cols))]

    X = data.values
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    prediction = model.predict(X).flatten()
    result = pd.DataFrame({'Instance':instanceIds, 'Income':prediction})
    result.to_csv(outputFile,index=False)


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
    plt.legend()
    plt.show()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.legend()
    plt.show()


"""
Steps:
read
make data consistent
populate missing
replace values
encode cat dat
scaling
split
fit
test
"""
if __name__ == '__main__':
    dataset = pd.read_csv('salaryPredTrain.csv')
    testDataset = pd.read_csv('salaryPredTest.csv')
    # dataset.profile_report(title='init data').to_file('pandasProfiles/initData.html')

    dataset = make_data_consistent(dataset)
    testDataset = make_data_consistent(testDataset)
    # dataset.profile_report(title='after making data consistent').to_file('pandasProfiles/consData.html')

    dataset = populate_missing_data(dataset)
    testDataset = populate_missing_data(testDataset)
    # dataset.profile_report(title='after populating missing data').to_file('pandasProfiles/popMissData.html')

    dataset = remove_rows(dataset)

    (dataset,testDataset) = encode_categories(dataset,testDataset)
    # countBefore = len(dataset)
    # dataset = dataset[(np.abs(stats.zscore(dataset[['Country','Profession']])) < 3).all(axis=1)]
    # countAfter = len(dataset)
    # print('Removed ' + str(countBefore - countAfter) + ' outliers from data')
    # dataset.profile_report(title='after encoding data').to_file('pandasProfiles/encData.html')
    print("Data pre-processing done")
    # plot_corr_matrix(dataset)

    # TODO:: feature selection optimise
    # featureColumns = ['University Degree','Profession','Country','Gender','Year of Record','Age','Body Height','Income in EUR']
    # featureColumns = ['Profession','Country','Year of Record','Age','Body Height','Income in EUR']
    featureColumns = np.array(dataset.columns.drop('Instance'))
    data = dataset.loc[:, dataset.columns.str.contains('|'.join(featureColumns))]
    X = data[data.columns.drop(['Income in EUR'])].values
    y = data['Income in EUR'].values
    (Xtrain, Xtest, ytrain, ytest) = split_data(X, y)
    print('Xtrain size:: ' + str(len(Xtrain)))
    print('Xtest size:: ' + str(len(Xtest)))

    # Scale Data
    scaler = StandardScaler()
    scaler.fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    # Scale Data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print("No of Columns:: " + str(len(X[0])))

    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(X[0])]),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        # layers.Dense(64, activation='relu'),
        # layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mae', 'mse'])

    print(model.summary())

    print("fitting model..")
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(Xtrain, ytrain, epochs=100, validation_split=0.2, verbose=0,callbacks=[early_stop,PrintDot()])
    print("fit model")

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())
    plot_history(history)

    rmse = np.sqrt(history.history['mean_squared_error'][-1])
    print('rmse:: ' + str(rmse))
    loss, mae, mse = model.evaluate(Xtest, ytest, verbose=2)
    print("Testing set Mean Sq Error: {:5.2f}".format(np.sqrt(mse)))

    dateTimeStamp = time.strftime('%Y%m%d%H%M%S')  # in the format YYYYMMDDHHMMSS
    predict_data(model,testDataset,featureColumns,'output/tf/tfOut-' + dateTimeStamp + "-" + str(int(rmse)) + ".csv")