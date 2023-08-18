import pandas as pd


def get_train_dates():
    train_dates = ['1989', '1994', '1999', '2004','2009', '2014', '2019', '2023']
    oos_dates = []
    for i in range(len(train_dates)-1):
        oos_dates.append([str(int(train_dates[i]) + 1), train_dates[i+1]])
    return train_dates, oos_dates


def load_features(name='ff3', start='1970'):
    filename = 'data/' + name + '.csv'
    x = pd.read_csv(filename, index_col=0, parse_dates=True)
    x = x[start:]
    x = x.iloc[:, :-1]
    x = to_weekly(x)
    #x = to_monthly(x)
    return x


def load_response(name='ind10', start='1970'):
    filename = 'data/' + name + '.csv'
    y = pd.read_csv(filename, index_col=0, parse_dates=True)
    y = y[start:]
    y = to_weekly(y)
    #y = to_monthly(y)
    return y


def to_weekly(x):
    return x.resample('W').sum()


def to_monthly(x):
    return x.resample('M').sum()



