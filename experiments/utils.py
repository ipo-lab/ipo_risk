import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_train_dates():
    train_dates = ['1989', '1994', '1999', '2004', '2009', '2014', '2019', '2023']
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


def plot_z_minvar(r_ols, r_ipo, n_samples=1000, sample_size=3*52, annual_factor=52):
    n = r_ols.shape[0]
    mat = np.zeros((n_samples, 2))
    for i in range(n_samples):
        idx = np.random.randint(n, size=sample_size)
        mat[i, 0] = r_ols[idx].var() * annual_factor
        mat[i, 1] = r_ipo[idx].var() * annual_factor

    idx = np.argsort(mat[:, 1])
    mat = mat[idx, :]
    dr = mat[:, 1] < mat[:, 0]
    dr = dr.mean()
    name = "IPO: DR={}".format(dr)
    mat = pd.DataFrame(mat, columns=["OLS", name])
    count = pd.DataFrame(np.arange(n_samples), columns=['count'])
    mat = pd.concat([mat, count],axis=1)
    # --- plot:
    fig, ax = plt.subplots()
    fig_ols = ax.scatter(x=mat['count'], y=mat['OLS'], color='lightseagreen')
    fig_ols.set_label('OLS')
    fig_ipo = ax.scatter(x=mat['count'], y=mat[name], color='darkorange')
    fig_ipo.set_label(name)
    ax.set_ylabel('Variance Cost', fontsize=14)
    ax.set_xlabel('Bootstrap Samples', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend()

    return None


def plot_z_maxdiv(r_ols, r_ipo, n_samples=1000, sample_size=3*52, annual_factor=52**0.5):
    n = r_ols.shape[0]
    mat = np.zeros((n_samples, 2))
    for i in range(n_samples):
        idx = np.random.randint(n, size=sample_size)
        mat[i, 0] = compute_maxdiv_loss(r_ols[idx]) * annual_factor
        mat[i, 1] = compute_maxdiv_loss(r_ipo[idx]) * annual_factor

    idx = np.argsort(mat[:, 1])
    mat = mat[idx, :]
    dr = mat[:, 1] < mat[:, 0]
    dr = dr.mean()
    name = "IPO: DR={}".format(dr)
    mat = pd.DataFrame(mat, columns=["OLS", name])
    count = pd.DataFrame(np.arange(n_samples), columns=['count'])
    mat = pd.concat([mat, count],axis=1)
    # --- plot:
    fig, ax = plt.subplots()
    fig_ols = ax.scatter(x=mat['count'], y=mat['OLS'], color='lightseagreen')
    fig_ols.set_label('OLS')
    fig_ipo = ax.scatter(x=mat['count'], y=mat[name], color='darkorange')
    fig_ipo.set_label(name)
    ax.set_ylabel('Diversification Cost', fontsize=14)
    ax.set_xlabel('Bootstrap Samples', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend()

    return None


def compute_maxdiv_loss(x):
    return -np.abs(x).mean() / x.std()


def plot_z_rp(r_ols, r_ipo, w_ols, w_ipo, n_samples=1000, sample_size=3*52):
    n = r_ols.shape[0]
    mat = np.zeros((n_samples, 2))
    for i in range(n_samples):
        idx = np.random.randint(n, size=sample_size)
        mat[i, 0] = compute_erc_loss(r_ols[idx], w_ols[idx])
        mat[i, 1] = compute_erc_loss(r_ipo[idx], w_ipo[idx])

    idx = np.argsort(mat[:, 1])
    mat = mat[idx, :]
    dr = mat[:, 1] < mat[:, 0]
    dr = dr.mean()
    name = "IPO: DR={}".format(dr)
    mat = pd.DataFrame(mat, columns=["OLS", name])
    count = pd.DataFrame(np.arange(n_samples), columns=['count'])
    mat = pd.concat([mat, count],axis=1)
    # --- plot:
    fig, ax = plt.subplots()
    fig_ols = ax.scatter(x=mat['count'], y=mat['OLS'], color='lightseagreen')
    fig_ols.set_label('OLS')
    fig_ipo = ax.scatter(x=mat['count'], y=mat[name], color='darkorange')
    fig_ipo.set_label(name)
    ax.set_ylabel('Risk Contribution Cost', fontsize=14)
    ax.set_xlabel('Bootstrap Samples', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend()

    return None


def compute_erc_loss(x, w):
    var = x.var()
    log_loss = np.log(w).sum(axis=1).mean()
    loss = 0.50 * var - log_loss
    return loss


def get_trade_weights(w, n_ahead=4, oos_start='1990'):
    w_smooth = w.rolling(n_ahead).mean()
    w_smooth = w_smooth[oos_start:]
    return w_smooth


def compute_returns(w, y_df, oos_start):
    w = w[oos_start:]
    y_df = y_df[oos_start:]
    out = pd.DataFrame(np.zeros((w.shape[0], 1)), index=w.index)
    r = (w.values * y_df)
    out[:] = r.values.sum(axis=1, keepdims=True)
    return out



