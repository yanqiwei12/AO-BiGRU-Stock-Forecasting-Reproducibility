# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Bidirectional
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from mealpy import AO
from mealpy import FloatVar, Problem
import time
start_time = time.time()
def xy_split(df, target):
    y_clmns=[target]
    x_clmns=df.columns.tolist()
    remove_clmns=[target]
    for arg in remove_clmns:
        x_clmns.remove(arg)
    X=df[x_clmns].iloc[:-1]
    y=df[y_clmns].iloc[1:]
    return X, y
data = pd.read_excel("DataSet_DowJones.xlsx", index_col='Date')
X, y = xy_split(data, 'Close')
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2, random_state=1)
X_train = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))
def decode_solution(solution):
    batch_size = 2**int(solution[0])
    epoch = 100 * int(solution[1])
    opt = 'Adam'
    learning_rate = solution[2]
    network_weight_initial = 'normal'
    activation = 'relu'
    n_hidden_units = 2**int(solution[3])
    return [batch_size, epoch, opt, learning_rate, network_weight_initial, activation, n_hidden_units]
def objective_function(solution): 
    batch_size, epoch, opt, learning_rate, network_weight_initial, Activation, n_hidden_units = decode_solution(solution)
    model = Sequential()
    model.add(Bidirectional(
              GRU(units = n_hidden_units, return_sequences=True), 
              input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    adam = Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam, loss='mse')
    model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, verbose=1)
    yhat = model(X_test)
    fitness = mean_absolute_error(y_test, yhat)
    return fitness
hyper_list = ['batch_size', 'epoch', 'learning_rate', 'n_hidden_units']
LB = [1, 1, 0.0001, 1]
UB = [4.99, 20.99, 1.0, 6.99]
problem = {
    "obj_func": objective_function,
    "bounds": FloatVar(lb=LB, ub=UB),
    "minmax": "min",
    "verbose": True,
    }
Epoch = 300
Pop_Size = 100
opt_Model = AO.OriginalAO(epoch=Epoch, pop_size=Pop_Size)
model_name = opt_Model.name
bests = opt_Model.solve(problem)
end_time = time.time()
print('Optimazing and prediction done in', round(end_time-start_time, 2), "secound")
batch_size, epoch, opt, learning_rate, network_weight_initial, Activation, n_hidden_units = decode_solution(bests.solution)
model = Sequential()
model.add(Bidirectional(
          GRU(units = n_hidden_units, return_sequences=True), 
          input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
adam = Adam(learning_rate=learning_rate)
model.compile(optimizer=adam, loss='mse')
model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, verbose=1)
predict = model(X_test)

#Backtesting
df = y_test.copy()
df['Prediction'] = predict
df = df.sort_values('Date').reset_index(drop=True)

initial_cash = 10000
cash = initial_cash
position = 0
portfolio = []
for i in range(len(df) - 1):
    current_price = df.loc[i, 'Close']
    predicted_next = df.loc[i, 'Prediction']
    next_price = df.loc[i+1, 'Close']
    if predicted_next > current_price:
        position = 1
    else:
        position = 0
    if i == 0:
        shares = cash / current_price if position == 1 else 0
    if position == 1 and (i == 0 or portfolio[-1]['position'] == 0):
        shares = cash / current_price
        cash = 0
    elif position == 0 and (i > 0 and portfolio[-1]['position'] == 1):
        cash = shares * current_price
        shares = 0
    port_value = shares * next_price + cash
    port_value = port_value * (1.0005)**i
    portfolio.append({'Date': df.loc[i+1, 'Date'], 'Portfolio': port_value, 'position': position})

pf = pd.DataFrame(portfolio)
bh_shares = initial_cash / df.loc[0, 'Close']
bh_values = bh_shares * df['Close'][1:].values
pf['BuyHold'] = bh_values

def cumulative_return(values):
    return (values[-1] / values[0]) - 1

def max_drawdown(values):
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    return np.max(drawdown)

def sharpe_ratio(values):
    returns = np.diff(values) / values[:-1]
    return (np.mean(returns) / np.std(returns, ddof=1)) * np.sqrt(252)

model_cum_ret = cumulative_return(pf['Portfolio'].values)
model_mdd = max_drawdown(pf['Portfolio'].values)
model_sharpe = sharpe_ratio(pf['Portfolio'].values)

bh_cum_ret = cumulative_return(pf['BuyHold'].values)
bh_mdd = max_drawdown(pf['BuyHold'].values)
bh_sharpe = sharpe_ratio(pf['BuyHold'].values)
results = pd.DataFrame({
    'Strategy': ['Model-Based', 'Buy & Hold'],
    'Cumulative Return (%)': [round(model_cum_ret*100, 2), round(bh_cum_ret*100, 2)],
    'Max Drawdown (%)': [round(model_mdd*100, 2), round(bh_mdd*100, 2)],
    'Sharpe Ratio': [round(model_sharpe, 2), round(bh_sharpe, 2)]
})
print(results)
