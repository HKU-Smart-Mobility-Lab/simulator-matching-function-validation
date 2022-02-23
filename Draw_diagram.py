from path import *
from Evaluate import *
import matplotlib.pyplot as plt
import re
import pandas as pd
import pickle

def get_best_model(result):
    model_error = []
    real_data = np.array(list(get_real_data(result).values()))
    model_error.append(list(abs(np.array(list(get_model_result_perfect_matching(result).values()))-real_data+np.array([0, 1000, 1000, 1000]))))
    model_error.append(list(abs(np.array(list(get_model_result_fcfs(result).values()))-real_data)))
    model_error.append(list(abs(np.array(list(get_model_result_production_function(result).values()))-real_data)))
    model_error.append(list(abs(np.array(list(get_model_result_mm1(result).values()))-real_data)))
    model_error.append(list(abs(np.array(list(get_model_result_mm1k(result).values()))-real_data)))
    model_error.append(list(abs(np.array(list(get_model_result_mmn(result).values()))-real_data)))
    model_error.append(list(abs(np.array(list(get_model_result_batch_matching(result).values()))-real_data)))
    model_error = np.array(model_error)
    best_model = np.argmin(model_error, axis=0)
    best_model_mape = model_error[best_model, list(np.arange(len(real_data)))]/real_data
    return best_model, best_model_mape

def get_model_name(labels):
    res = []
    for la in labels:
        model_num = int(re.split(r'[{}]',la)[1])
        res.append(model_list[model_num])
    return res


def draw_best_model(x, y, labels, errors, name):
    fs = 20

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)

    t = trip_time
    x_stack = np.array(
        [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
    y1 = np.array(
        [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
    y2 = x_stack * t / 1 - y1
    y3 = x_stack * t / 0.7 - y1 - y2
    y5 = np.array(
        [2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000,
         2000, 2000, 2000]) - y1 - y2 - y3
    y_stack = [y1, y2, y3, y5]
    plt.stackplot(x_stack, y_stack, colors=['w', "w", "#808080", "#016795"], alpha=0.08)
    ax.plot(x_stack, x_stack * t / 1, linestyle='-', color="red", lw=1, label='$Q = N/t$', alpha=0.5)
    scatter = ax.scatter(x, y, c=labels, cmap='Dark2', s=errors * 80 + 10)

    handles, labels = scatter.legend_elements(prop="colors", alpha=1.0)

    labels = get_model_name(labels)

    legend1 = ax.legend(handles, labels, loc=2, bbox_to_anchor=(1.01, 1.0), title="Models", fontsize=fs)
    legend2 = ax.legend(loc=2, bbox_to_anchor=(1.01, 0.6), markerscale=0.5, fontsize=fs)
    plt.setp(legend1.get_title(), fontsize=fs)
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)

    kw = dict(prop="sizes", fmt="{x:.0f}%", alpha=0.6,
              func=lambda s: ((s - 10) / 0.8))
    legend2 = ax.legend(*scatter.legend_elements(**kw),
                        loc=2, bbox_to_anchor=(1.01, 0.5), title="MAPE", fontsize=fs)
    plt.setp(legend2.get_title(), fontsize=fs)
    plt.xlabel('Arrival rate of orders (pax/sec)', fontsize=20)

    plt.ylabel('Number of drivers (veh)', fontsize=20)

    plt.savefig(load_path + 'Figures/' + name + '.jpg', dpi=600, bbox_inches='tight')
    plt.show()
    plt.clf()

def get_file_list(num, mode):
    if mode == 'fix_driver':
        para_select = 3
        para_sort = 1
    elif mode == 'fix_order':
        para_select= 1
        para_sort = 3
    file_list = []
    for file in files:
        if float(file.split('_')[para_select]) == num:
            file_list.append(file)
    file_list.sort(key=lambda ele:float(re.split(r'[____]',ele)[para_sort]))
    return file_list

def get_data(result):
    res = []
    res.append(list(get_real_data(result).values()))
    res.append(list(get_model_result_perfect_matching(result).values()))
    res.append(list(get_model_result_fcfs(result).values()))
    res.append(list(get_model_result_production_function(result).values()))
    res.append(list(get_model_result_mm1(result).values()))
    res.append(list(get_model_result_mm1k(result).values()))
    res.append(list(get_model_result_mmn(result).values()))
    res.append(list(get_model_result_batch_matching(result).values()))
    res = np.array(res).T.tolist()
    return res

def filter_line(x, y):
    new_x  = []
    new_y = []
    for i in range(len(y)):
        if y[i]>=0:
            new_x.append(x[i])
            new_y.append(y[i])
    return new_x, new_y

def get_draw_data(y, name, mode):
    location = "lower right"
    model_list = ['Real data', 'Perfect Matching','FCFS','Cobb-Douglas Production Function','M/M/1 Queuing Model','M/M/1/k Queuing Model','M/M/N Queuing Model',  'Batch Matching']
    if name == 'Matching rate':
        if mode == 'fix_driver':
            location = "upper left"
        model_list = ['Real data', 'Perfect Matching','FCFS','Cobb-Douglas Production Function','M/M/1 and M/M/N Queuing Model','M/M/1/k Queuing Model',  'Batch Matching']
        y = pd.DataFrame(y)[[0, 1, 2, 3, 4, 5, 7]].values.tolist()
    if name == 'Pick-up time':
        model_list = ['Real data','Cobb-Douglas and M/M/1/k','FCFS','M/M/1, M/M/N Queuing Model',  'Batch Matching']
        # df = pd.DataFrame(y)
        y = pd.DataFrame(y)[[0, 1, 2, 4, 7]].values.tolist()
        if mode == 'fix_driver':
            location = "upper left"
        else:
            location = "upper right"
    if name == 'Matching time' or name == 'Waiting time':
        model_list = ['Real data', 'FCFS','Cobb-Douglas Production Function','M/M/1 Queuing Model','M/M/1/k Queuing Model','M/M/N Queuing Model',  'Batch Matching']
        y = pd.DataFrame(y)[[0, 2, 3, 4, 5, 6, 7]].values.tolist()
        if mode == 'fix_driver':
            location = "upper left"
        else:
            location = "upper right"
    return y, model_list, location


def draw_picture(x, y, name, mode, num):
    fs = 20
    y, model_list, location = get_draw_data(y, name, mode)
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 10)
    if mode == 'fix_driver':
        ax.axvspan(0, 1.17, facecolor="#016795", alpha=0.08)
        ax.axvspan(1.17, 1.67, facecolor="#808080", alpha=0.08)
        x_label = 'Arrival rate of orders (pax/sec)'
    elif mode == 'fix_order':
        ax.axvspan(600, 998, facecolor="#808080", alpha=0.08)
        ax.axvspan(998, 2000, facecolor="#016795", alpha=0.08)
        x_label = 'Number of drivers (veh)'

    for i in range(len(y[0])):
        current_line = np.array(y)[:, i:i + 1].T[0]
        if i == 0:
            m = '*'
            m_size = 10
            l_width = 2
        else:
            m = 'o'
            m_size = 5
            l_width = 1
        new_x, new_y = filter_line(x, current_line)
        l = plt.plot(new_x, new_y, label=model_list[i], marker=m, linewidth=l_width, markersize=m_size)
    legend1 = plt.legend(loc=location, title="Models", fontsize=fs)

    plt.setp(legend1.get_title(), fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)

    plt.xlabel(x_label, fontsize=fs)
    if name == 'Matching rate':
        y_label_name = 'Matching rate (pax/sec)'
    else:
        y_label_name = name + ' (sec)'

    plt.ylabel(y_label_name, fontsize=fs)
    picture_name = name + '_' + mode + '=' + str(num)

    plt.savefig(load_path + 'Figures/' + picture_name + '.jpg', dpi=600, bbox_inches='tight')
    plt.show()
    plt.cla()
    plt.close("all")

def draw_one_picture(mode, num, trip_time):
    file_list = get_file_list(num, mode)
    x = []
    matching_rate = []
    matching_time = []
    pickup_time = []
    waiting_time = []
    if mode == 'fix_driver':
        para_x = 1
    elif mode == 'fix_order':
        para_x = 3
    for file in file_list:
        result = pickle.load(open(result_path + file, 'rb'))
        if float(file.split('_')[1])/2 > result['fleet_size']/trip_time:
            pass
        else:
            x.append(float(file.split('_')[para_x]))
            res_data = get_data(result)
            matching_rate.append(res_data[0])
            matching_time.append(res_data[1])
            pickup_time.append(res_data[2])
            waiting_time.append(res_data[3])
    if mode == 'fix_driver':
        x=np.array(x)/2
    draw_picture(x, matching_rate, 'Matching rate', mode, num)
    draw_picture(x, matching_time, 'Matching time', mode, num)
    draw_picture(x, pickup_time, 'Pick-up time', mode, num)
    draw_picture(x, waiting_time, 'Waiting time', mode, num)

if __name__ == "__main__":
    plt.rc('font', family='Times New Roman')
    production_func_params = get_production_func_params()
    model_list = ['Perfect Matching','FCFS','Cobb-Douglas Production Function','M/M/1 Queuing Model', 'M/M/1/k Queuing Model', 'M/M/N Queuing Model','Batch Matching']

    files= os.listdir(result_path)
    files.sort()
    time = []
    for file in files:
        result = pickle.load(open(result_path + file, 'rb'))
        time.append(result['trip_time'])
    trip_time = int(np.array(time).mean())

    orders = []
    drivers = []
    best_model = []
    best_model_mape = []
    for file in files:
        result = pickle.load(open(result_path + file, 'rb'))
        if float(file.split('_')[1])/2 > result['fleet_size']/trip_time:
            pass
        else:
            orders.append(float(file.split('_')[1]))
            drivers.append(float(file.split('_')[3]))
            m, m_e = get_best_model(result)
            best_model.append(list(m))
            best_model_mape.append(list(m_e))
    x=np.array(orders)/2
    y=np.array(drivers)

    labels=np.array(np.array(best_model)[:,:1].T)[0]
    errors = np.array(np.array(best_model_mape)[:,:1].T)[0]
    draw_best_model(x, y, labels,errors, 'matching_rate_best_model')
    labels=np.array(np.array(best_model)[:,1:2].T)[0]
    errors = np.array(np.array(best_model_mape)[:,1:2].T)[0]
    draw_best_model(x, y, labels, errors,'matching_time_best_model')
    labels=np.array(np.array(best_model)[:,2:3].T)[0]
    errors = np.array(np.array(best_model_mape)[:,2:3].T)[0]
    draw_best_model(x, y, labels, errors, 'pickup_time_best_model')
    labels=np.array(np.array(best_model)[:,3:4].T)[0]
    errors = np.array(np.array(best_model_mape)[:,3:4].T)[0]
    draw_best_model(x, y, labels, errors, 'waiting_time_best_model')

    draw_one_picture('fix_driver', 1000, trip_time)
    draw_one_picture('fix_order', 2.0, trip_time)