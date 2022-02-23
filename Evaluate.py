import math
from mpmath import mp, mpf, gamma
import os
import pickle
import numpy as np
from scipy import optimize
from scipy.optimize import least_squares

result_path = 'Results/'
area=10000*10000
speed=11.11

def get_real_data(result):
    res = {}
    res['matching_rate'] = result['matching_rate']
    res['matching_time'] = result['matching_time']
    res['pickup_time'] = result['pickup_time']
    res['waiting_time'] = result['effective_orders_total_waiting_time']
    return res

def get_model_result_perfect_matching(result):
    Q = result['total_requests']/result['total_time']
    Tv = result['fleet_size']/result['trip_time']
    model_res = {}
    model_res['matching_rate'] = min(Q, Tv)
    model_res['matching_time'] = 0
    model_res['pickup_time'] = 0
    model_res['waiting_time'] = 0
    return model_res

def get_model_result_mm1(result):
    Q = result['total_requests']/result['total_time']
    Tv = result['fleet_size']/result['trip_time']
    model_res = {}
    if Q>=Tv:
        model_res['matching_rate'] = -1
        model_res['matching_time'] = -1
        model_res['pickup_time'] = -1
        model_res['waiting_time'] = -1
    else:
        model_res['matching_rate']  = Q
        model_res['matching_time'] = Q/(Tv*(Tv-Q))
        model_res['pickup_time'] = 0
        model_res['waiting_time'] = model_res['matching_time']
    return model_res

def get_model_result_mm1k(result):
    mp.dps = 300
    k = result['max_waiting_orders']
    Q = result['total_requests']/result['total_time'] # lambda
    Tv = result['fleet_size']/result['trip_time'] # miu
    r = Q/Tv
    model_res = {}

    P0 = (1-mpf(r))/(1-pow(mpf(r), k+1))
    Pn = ((1-mpf(r))/(1-pow(mpf(r), k+1)))*pow(mpf(r), k)
    Ls = mpf(r)/(1-mpf(r))-((k+1)*pow(mpf(r), k+1))/(1-pow(mpf(r), k+1))
    Lq = Ls -1 +P0
    model_res['matching_rate'] = round(Q*(1-Pn), 6)
    model_res['matching_time'] = round(Lq/(Q*(1-Pn)), 2)
    model_res['pickup_time'] = 0
    model_res['waiting_time'] = model_res['matching_time']
    return model_res


def equations_of_ns_market_FCFS(Var, *decision_variables):
    Q, N, A, t, d_ratio, v = decision_variables
    Nv, wp = Var
    e1 = N - Nv - Q * t - Q * wp
    e2 = wp - d_ratio /(2 * v * ((Nv/A) ** 0.5))
    return e1, e2

def get_model_result_fcfs(result):
    Q = result['total_requests']/result['total_time']
    t = result['trip_time']
    N = result['fleet_size']
    v = result['speed']
    d_ratio = 1.27

    decision_variables = (Q, N, area, t, d_ratio, v)
    res = least_squares(equations_of_ns_market_FCFS, (N / 2, 0), bounds=((0, 0), (N, 100000)),
                        args=decision_variables)

    model_res = {}

    if abs(res.fun[0]) > 1: # 无解
        model_res['matching_rate'] = -1
        model_res['matching_time'] = -1
        model_res['pickup_time'] = -1
        model_res['waiting_time'] = -1
    else:
        model_res['matching_rate'] = result['total_requests']/result['total_time']
        model_res['matching_time'] = 0
        model_res['pickup_time'] = res.x[1]
        model_res['waiting_time'] = model_res['pickup_time']
    return model_res

def cal_pickup_time_for_batch_matching(Nc, Nv, speed):
    a = 1.27/(1- np.exp(-Nc/Nv))
    b = 1/(2*pow(Nc/area, 0.5))
    c = math.erf(pow(Nc/Nv, 0.5))
    d = 1/pow(np.pi*Nv/area,0.5)
    e = np.exp(-Nc/Nv)
    res = a*(b*c - d*e)
    res = res/speed
    return res


def equations_of_ns_market_BM(Var, *decision_variables):

    Q, N, t = decision_variables
    wt, wm, wp = Var

    Nv = Q * wt
    Nc = Q * wm
    matching_time = 2 / ((Nv / Nc) * (1 - np.exp(-Nc / Nv)))
    pickup_time = cal_pickup_time_for_batch_matching(Nc, Nv, speed)

    e1 = N - Q * (wt + wp + t)
    e2 = wm - matching_time
    e3 = wp - pickup_time

    return e1, e2, e3


def get_model_result_batch_matching(result):
    Q = result['total_requests']/result['total_time']
    N = result['fleet_size']
    t = result['trip_time']
    decision_variables = (Q, N, t)
    res = least_squares(equations_of_ns_market_BM, (10, 10, 10),bounds=((0, 0, 0), (500000000, 100000, 100000)),
                        args=decision_variables)
    model_res = {}
    if abs(res.fun[0]) > 10: # 无解
        model_res['matching_rate'] = -1
        model_res['matching_time'] = -1
        model_res['pickup_time'] = -1
        model_res['waiting_time'] = -1
    else:
        model_res['matching_rate'] = result['total_requests']/result['total_time']
        model_res['matching_time'] = res.x[1]
        model_res['pickup_time'] = res.x[2]
        model_res['waiting_time'] = model_res['matching_time']+model_res['pickup_time']
    return model_res

def cal_matching_time_for_mmn(result):
    mp.dps = 300

    lamda = result['total_requests']/result['total_time'] # Q
    u = 1/result['trip_time']
    c = result['fleet_size']
    r = lamda / u
    rou = r/c

    item_1 = 0

    for i in range(c):
        item_1 += mpf(r) ** mpf(i) / mpf(np.math.factorial(i))
    item_2 = mpf(r) ** mpf(c) / gamma(c+1) / (1 - rou)
    prob_0 = 1 / (item_1 + item_2)
    queueing_time = round(prob_0 * mpf(r)**(c) / gamma(c+1) / (c*u) / (1-rou)**2, 10)
    return queueing_time

def get_model_result_mmn(result):
    model_res = {}
    if result['total_requests']/result['total_time']>=result['fleet_size']/result['trip_time']:
        model_res['matching_rate'] = -1
        model_res['matching_time'] = -1
        model_res['pickup_time'] = -1
        model_res['waiting_time'] = -1
    else:
        model_res['matching_rate'] = result['total_requests']/result['total_time']
        model_res['matching_time'] = cal_matching_time_for_mmn(result)
        model_res['pickup_time'] = 0
        model_res['waiting_time'] = model_res['matching_time']
    return model_res

def linear_func(ln_Nv, ln_Nc, p):
    a, b, c = p
    return a * ln_Nv + b * ln_Nc + c


def residuals(p, y, ln_Nv, ln_Nc):
    return y - linear_func(ln_Nv, ln_Nc, p)


def get_production_func_params():
    files = os.listdir(result_path)
    collected_data = []
    for file in files:
        f = pickle.load(open(result_path + file, 'rb'))
        collected_data.append([f['vacant_vehicles'], f['mean_waiting_orders'], f['total_requests'] /f['total_time']])
    ln_collected_data = np.log(collected_data)
    ln_Nv = np.array(ln_collected_data[:, :1]).T[0]
    ln_Nc = np.array(ln_collected_data[:, 1:2]).T[0]
    y = np.array(ln_collected_data[:, 2:3]).T[0]

    func_param = optimize.leastsq(residuals, np.array([0, 0, 0]), args=(y, ln_Nv, ln_Nc))[0]
    func_param[2] = np.exp(func_param[2])
    return func_param

def get_model_result_production_function(result):
    Q = result['total_requests']/result['total_time']
    N = result['fleet_size']
    t = result['trip_time']
    decision_variables = (Q, N, t)
    res = least_squares(equations_of_ns_market_CD, (N / 2, 0), bounds=((0, 0), (N+1, 1000)),
                        args=decision_variables)
    model_res = {}
    if abs(res.fun[1]) > 0.01: # 无解
        model_res['matching_rate'] = -1
        model_res['matching_time'] = -1
        model_res['pickup_time'] = -1
        model_res['waiting_time'] = -1
    else:
        model_res['matching_rate'] = Q
        model_res['matching_time'] = res.x[1]
        model_res['pickup_time'] = 0
        model_res['waiting_time'] = model_res['matching_time']
    return model_res


def equations_of_ns_market_CD(Var, *decision_variables):
    production_func_params = get_production_func_params()
    alpha1, alpha2, A = production_func_params

    Q, N, t= decision_variables
    Nv, wm = Var

    e1 = N - Nv - Q * t
    e2 = Q - A * (Nv ** alpha1) * ((Q * wm) ** alpha2)

    return e1, e2


def get_fcfs_params():
    files = os.listdir(result_path)
    collected_data = []
    for file in files:
        f = pickle.load(open(result_path + file, 'rb'))
        collected_data.append([f['vacant_vehicles']*f['vacant_vehicles'], f['pickup_time']])
    ln_collected_data = np.log(collected_data)
    x = np.array(ln_collected_data[:, :1]).T[0]
    y = np.array(ln_collected_data[:, 1:2]).T[0]

    func_param = optimize.leastsq(residuals_fcfs, np.array([0]), args=(y, x))[0][0]
    func_param = np.exp(func_param)
    return func_param

def fcfs_func(x, c):
    return -x+c


def residuals_fcfs(c, y, x):
    return y - fcfs_func(x, c)
