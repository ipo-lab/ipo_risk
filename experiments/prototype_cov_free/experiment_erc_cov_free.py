import matplotlib.pyplot as plt
import torch
from ts.loss import loss_erc
from experiments.utils import load_response, load_features, get_train_dates, plot_z_rp, get_trade_weights, compute_returns, optimize_model
from ts.popt import RPCCC, RPCCCOLS, RPDCC, RPDCCOLS


# --- load datasets:
feature = 'ff3'
x_df = load_features(feature)
y_df = load_response('ind10')
train_dates, oos_dates = get_train_dates()

# --- init weights:
weights_ccc = y_df * 0
weights_dcc = y_df * 0
weights_ccc_ols = y_df * 0
weights_dcc_ols = y_df * 0

# --- params
lr = 0.25
n_epochs = 500
annual_factor = 52**0.5
n_ahead = 1
n_starts = 1

torch.manual_seed(0)
# --- main loop
for i in range(len(oos_dates)):
    print('training iteration: {}'.format(i))
    # --- prep data:
    train_dates_i = train_dates[i]
    oos_dates_i = oos_dates[i]
    oos_dates_start = oos_dates_i[0]
    oos_dates_end = oos_dates_i[1]
    x_is = torch.as_tensor(x_df[:train_dates_i].values)
    y_is = torch.as_tensor(y_df[:train_dates_i].values)
    x_oos = torch.as_tensor(x_df[:oos_dates_end].values)
    y_oos = torch.as_tensor(y_df[:oos_dates_end].values)
    idx_oos = slice(x_df.index.get_loc(oos_dates_start).start, x_df.index.get_loc(oos_dates_end).stop)

    # --- Train OLS Models: on init:
    model_ccc_ols = RPCCCOLS(x=x_is, y=y_is, n_ahead=n_ahead)
    model_dcc_ols = RPDCCOLS(x=x_is, y=y_is, n_ahead=n_ahead)

    # --- Train IPO Models: CCC
    model_ccc = optimize_model(x=x_is, y=y_is, model_init=RPCCC, loss_fn=loss_erc,
                               lr=lr, n_epochs=n_epochs, n_starts=n_starts, n_ahead=n_ahead, init='random')
    # --- Train IPO Models: CDCC
    model_dcc = optimize_model(x=x_is, y=y_is, model_init=RPDCC, loss_fn=loss_erc,
                               lr=lr, n_epochs=n_epochs, n_starts=n_starts, n_ahead=n_ahead, init='random')

    # --- out-of-sample:
    weights_ccc_ols[idx_oos] = model_ccc_ols(x=x_oos).squeeze(2).detach().numpy()[idx_oos]
    weights_dcc_ols[idx_oos] = model_dcc_ols(x=x_oos).squeeze(2).detach().numpy()[idx_oos]
    weights_ccc[idx_oos] = model_ccc(x=x_oos).squeeze(2).detach().numpy()[idx_oos]
    weights_dcc[idx_oos] = model_dcc(x=x_oos).squeeze(2).detach().numpy()[idx_oos]


# --- oos evaluation:
oos_start = oos_dates[0][0]
# --- CCC OLS:
w_ccc_ols = get_trade_weights(w=weights_ccc_ols, oos_start=oos_start, n_ahead=n_ahead)
r_ccc_ols = compute_returns(w=w_ccc_ols, y_df=y_df, oos_start=oos_start)

# --- DCC OLS
w_dcc_ols = get_trade_weights(w=weights_dcc_ols, oos_start=oos_start, n_ahead=n_ahead)
r_dcc_ols = compute_returns(w=w_dcc_ols, y_df=y_df, oos_start=oos_start)

# --- CCC IPO
w_ccc_ipo = get_trade_weights(w=weights_ccc, oos_start=oos_start, n_ahead=n_ahead)
r_ccc_ipo = compute_returns(w=w_ccc_ipo, y_df=y_df, oos_start=oos_start)

# --- DCC IPO
w_dcc_ipo = get_trade_weights(w=weights_dcc, oos_start=oos_start, n_ahead=n_ahead)
r_dcc_ipo = compute_returns(w=w_dcc_ipo, y_df=y_df, oos_start=oos_start)

# --- main plots:
out_dir = 'images/'
# --- CCC:
plot_z_rp(r_ols=r_ccc_ols.values,r_ipo=r_ccc_ipo.values, w_ols=w_ccc_ols.values,w_ipo=w_ccc_ipo.values)

# --- DCC:
plot_z_rp(r_ols=r_dcc_ols.values, r_ipo=r_dcc_ipo.values, w_ols=w_dcc_ols.values, w_ipo=w_dcc_ipo.values)


# --- Equity Plot:
# --- CCC:
r_ccc_ols_norm = r_ccc_ols/w_ccc_ols.values.sum(axis=1, keepdims=True)
r_ccc_ipo_norm = r_ccc_ipo/w_ccc_ipo.values.sum(axis=1, keepdims=True)
plt.plot(r_ccc_ols_norm.cumsum()/100, color='lightseagreen')
plt.plot(r_ccc_ipo_norm.cumsum()/100, color='darkorange')
plt.legend(["OLS", "IPO"])


# --- DCC:
r_dcc_ols_norm = r_dcc_ols/w_dcc_ols.values.sum(axis=1, keepdims=True)
r_dcc_ipo_norm = r_dcc_ipo/w_dcc_ipo.values.sum(axis=1, keepdims=True)
plt.plot(r_dcc_ols_norm.cumsum()/100, color='lightseagreen')
plt.plot(r_dcc_ipo_norm.cumsum()/100, color='darkorange')
plt.legend(["OLS", "IPO"])

