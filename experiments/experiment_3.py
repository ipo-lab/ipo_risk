import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ts.loss import loss_erc
from experiments.utils import load_response, load_features, get_train_dates, plot_z_rp
from ts.popt import RPCCC, RPCCCOLS, RPDCC, RPDCCOLS


# --- load datasets:
x_df = load_features('ff3')
y_df = load_response('ind10')
train_dates, oos_dates = get_train_dates()

# --- init weights:
weights_ccc = y_df * 0
weights_dcc = y_df * 0
weights_ccc_ols = y_df * 0
weights_dcc_ols = y_df * 0

# --- params
lr = 0.10
n_epochs = 100
annual_factor = 52**0.5
n_ahead = 4

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
    model_ccc = RPCCC(x=x_is, y=y_is, n_ahead=n_ahead)
    optimizer = torch.optim.Adam(model_ccc.parameters(), lr=lr)
    # ---- main training loop
    loss_hist = []
    for epoch in range(n_epochs):
        z = model_ccc(x=x_is)
        loss = loss_erc(z.squeeze(2), y_is)
        loss_hist.append(loss.item())
        optimizer.zero_grad()
        # --- compute gradients
        loss.backward()
        # --- update parameters
        optimizer.step()
        print('epoch {}, loss {}'.format(epoch, loss.item()))
    #
    # --- Train IPO Models: CCC
    model_dcc = RPDCC(x=x_is, y=y_is, n_ahead=n_ahead)
    optimizer = torch.optim.Adam(model_dcc.parameters(), lr=lr)
    # ---- main training loop
    loss_hist = []
    for epoch in range(n_epochs):
        z = model_dcc(x=x_is)
        loss = loss_erc(z.squeeze(2), y_is)
        loss_hist.append(loss.item())
        optimizer.zero_grad()
        # --- compute gradients
        loss.backward()
        # --- update parameters
        optimizer.step()
        print('epoch {}, loss {}'.format(epoch, loss.item()))

    # --- out-of-sample:
    weights_ccc_ols[idx_oos] = model_ccc_ols(x=x_oos).squeeze(2).detach().numpy()[idx_oos]
    weights_dcc_ols[idx_oos] = model_dcc_ols(x=x_oos).squeeze(2).detach().numpy()[idx_oos]
    weights_ccc[idx_oos] = model_ccc(x=x_oos).squeeze(2).detach().numpy()[idx_oos]
    weights_dcc[idx_oos] = model_dcc(x=x_oos).squeeze(2).detach().numpy()[idx_oos]


# --- oos evaluation:
oos_start = oos_dates[0][0]
weights_ccc_ols_smooth = weights_ccc_ols*1.0#.div(weights_ccc_ols.sum(axis=1) + 1e-12, axis=0)
weights_ccc_ols_smooth = weights_ccc_ols_smooth.rolling(n_ahead).mean()
p_ccc_ols = (weights_ccc_ols_smooth.values * y_df)[oos_start:]
p_ccc_ols = p_ccc_ols.values.sum(axis=1)
p_ccc_ols.std()/100*annual_factor

weights_dcc_ols_smooth = weights_dcc_ols*1.0#.div(weights_dcc_ols.sum(axis=1) + 1e-12, axis=0)
weights_dcc_ols_smooth = weights_dcc_ols_smooth.rolling(n_ahead).mean()
p_dcc_ols = (weights_dcc_ols_smooth * y_df)[oos_start:]
p_dcc_ols = p_dcc_ols.values.sum(axis=1)
p_dcc_ols.std()/100*annual_factor

weights_ccc_smooth = weights_ccc*1.0#.div(weights_ccc.sum(axis=1) + 1e-12, axis=0)
weights_ccc_smooth = weights_ccc_smooth.rolling(n_ahead).mean()
p_ccc = (weights_ccc_smooth.values * y_df)[oos_start:]
p_ccc = p_ccc.values.sum(axis=1)
p_ccc.std()/100*annual_factor

weights_dcc_smooth = weights_dcc*1.0#.div(weights_dcc.sum(axis=1) + 1e-12, axis=0)
weights_dcc_smooth = weights_dcc_smooth.rolling(n_ahead).mean()
p_dcc = (weights_dcc_smooth * y_df)[oos_start:]
p_dcc = p_dcc.values.sum(axis=1)
p_dcc.std()/100*annual_factor

# --- main plots:
w_ols = weights_ccc_ols.rolling(n_ahead).mean()[oos_start:].values
w_ipo = weights_ccc.rolling(n_ahead).mean()[oos_start:].values
plot_z_rp(p_ols=p_ccc_ols, p_ipo=p_ccc, w_ols=w_ols, w_ipo=w_ipo)

w_ols = weights_dcc_ols.rolling(n_ahead).mean()[oos_start:].values
w_ipo = weights_dcc.rolling(n_ahead).mean()[oos_start:].values
plot_z_rp(p_ols=p_dcc_ols, p_ipo=p_dcc, w_ols=w_ols, w_ipo=w_ipo)

plt.plot(p_ccc_ols.cumsum())
plt.plot(p_ccc.cumsum())
plt.plot(p_dcc_ols.cumsum())
plt.plot(p_dcc.cumsum())
p_ccc_ols.mean()/p_ccc_ols.std()
p_dcc_ols.mean()/p_dcc_ols.std()
p_ccc.mean()/p_ccc.std()
p_dcc.mean()/p_dcc.std()