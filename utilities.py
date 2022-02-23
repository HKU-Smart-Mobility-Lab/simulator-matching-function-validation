import numpy as np
import pandas as pd
from copy import deepcopy
from dispatch_alg import dispatch_alg_array
from path import *
import time
import pickle


def distance(coord_1, coord_2):
    x1, y1 = coord_1
    x2, y2 = coord_2
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    manhattan_dis = dx + dy

    return manhattan_dis

def distance_array(coord_1, coord_2):
    coord_1 = coord_1.astype(float)
    coord_2 = coord_2.astype(float)

    dx = np.abs(coord_2[:, 0] - coord_1[:, 0])
    dy = np.abs(coord_2[:, 1] - coord_1[:, 1])

    manhattan_dis = dx + dy

    return manhattan_dis

def cruise(row):
    if row['direction'] ==0:
        return [1, 0]
    elif row['direction'] ==1:
        return [-1, 0]
    elif row['direction'] ==2:
        return [0, 1]
    else:
        return [0, -1]

def cal_grid_id(target, side_length, grid_length):
    grid_x = target[:, :1] // grid_length
    grid_y = target[:, 1:] // grid_length
    return grid_y * side_length // grid_length + grid_x


class GridSystem:
    def __init__(self, **kwargs):
        #read parameters
        self.side_length = kwargs.pop('grid_system_side_length', 10000)
        self.grid_length = kwargs.pop('grid_system_grid_length', 1000)
        self.num_grid = int(self.side_length//self.grid_length)

    def create_grid(self):
        zone_id = np.arange(0, self.num_grid * self.num_grid)
        self.df_neighbor_centroid = pd.DataFrame(zone_id, columns=["zone_id"])
        self.df_neighbor_centroid['centroid_lng'] = (self.df_neighbor_centroid['zone_id']%self.num_grid)*self.grid_length+self.grid_length/2
        self.df_neighbor_centroid['centroid_lat'] = (self.df_neighbor_centroid['zone_id']//self.num_grid)*self.grid_length+self.grid_length/2

    def get_matrix(self):
        matrix = [[] for _ in range(self.num_grid)]
        for i in range(self.num_grid - 1, -1, -1):
            for j in range((self.num_grid - i - 1) * self.num_grid, (self.num_grid - i - 1) * self.num_grid + self.num_grid):
                matrix[i].append(j)
        return matrix

    def create_adj_mat(self):
        matrix = self.get_matrix()
        n = self.num_grid
        adj_mat = np.identity(n * n)
        for i in range(n):
            for j in range(n):
                for (x, y) in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                    if 0 <= x < n and 0 <= y < n:
                        adj_mat[matrix[i][j]][matrix[x][y]] = 1
        return adj_mat

    def load_data(self):
        self.create_grid()
        self.total_grid = self.df_neighbor_centroid.shape[0]
        self.adj_mat = self.create_adj_mat()

    def get_basics(self):
        #output: basic information about the grid network
        return self.total_grid


def sample_orders_drivers(request_distribution, driver_info, request_databases, driver_number_dist):
    # Used to refine the data
    sampled_request_distribution = request_distribution
    sampled_driver_info = driver_info
    sampled_request_databases = request_databases
    sampled_driver_number_dist = driver_number_dist
    return sampled_request_distribution, sampled_driver_info, sampled_request_databases, sampled_driver_number_dist


def sample_all_drivers(driver_info, t_initial, t_end, driver_sample_ratio=1, driver_number_dist=''):
    # generate all the drivers in the system
    # add states for vehicle_tabe based on vehicle columns
    # the driver info inlcud: 'driver_id', 'start_time', 'end_time', 'lng', 'lat'
    # the new driver info include: 'driver_id', 'start_time', 'end_time', 'lng', 'lat', 'status',
    # 'target_lng','target_lat','remaining_time', 'matched_order_id', 'total_idle_time'
    # t_initial and t_end are used to filter available drivers in the selected time period
    new_driver_info = deepcopy(driver_info)
    sampled_driver_info = new_driver_info

    sampled_driver_info['status'] = 0
    sampled_driver_info['target_loc_lng'] = sampled_driver_info['lng']
    sampled_driver_info['target_loc_lat'] = sampled_driver_info['lat']
    sampled_driver_info['target_grid_id'] = sampled_driver_info['grid_id']
    sampled_driver_info['remaining_time'] = 0
    sampled_driver_info['matched_order_id'] = 'None'
    sampled_driver_info['total_idle_time'] = 0

    return sampled_driver_info


def KM_simulation(wait_requests, driver_table, method = 'nothing', pick_up_dis_threshold = 950):
    # currently, we use the dispatch alg of peibo
    idle_driver_table = driver_table[driver_table['status'] == 0]
    num_wait_request = wait_requests.shape[0]
    num_idle_driver = idle_driver_table.shape[0]

    if num_wait_request > 0 and num_idle_driver > 0:
        starttime_1 = time.time()

        request_array = wait_requests.loc[:, ['origin_lng', 'origin_lat', 'order_id', 'weight']].values
        request_array = np.repeat(request_array, num_idle_driver, axis=0)
        driver_loc_array = idle_driver_table.loc[:, ['lng', 'lat', 'driver_id']].values
        driver_loc_array = np.tile(driver_loc_array, (num_wait_request, 1))
        assert driver_loc_array.shape[0] == request_array.shape[0]
        dis_array = distance_array(request_array[:, :2], driver_loc_array[:, :2])
        # print('negative: ', np.where(dis_array)<0)

        flag = np.where(dis_array <= pick_up_dis_threshold)[0]
        if method == 'pickup_distance':
            order_driver_pair = np.vstack([request_array[flag, 2], driver_loc_array[flag, 2], pick_up_dis_threshold + 1 - dis_array[flag], dis_array[flag]]).T
        else:
            order_driver_pair = np.vstack([request_array[flag, 2], driver_loc_array[flag, 2], request_array[flag, 3], dis_array[flag]]).T
        order_driver_pair = order_driver_pair.tolist()

        endtime_1 = time.time()
        dtime_1 = endtime_1 - starttime_1

        if len(order_driver_pair) > 0:
            #matched_pair_actual_indexs = km.run_kuhn_munkres(order_driver_pair)
            matched_pair_actual_indexs = dispatch_alg_array(order_driver_pair)

            endtime_2 = time.time()
            dtime_2 = endtime_2 - endtime_1
        else:
            matched_pair_actual_indexs = []
    else:
        matched_pair_actual_indexs = []

    return matched_pair_actual_indexs

def save_data(simulator, env_params):
    final_record_df = pd.DataFrame(simulator.requests_final_records)
    final_record = {}
    final_record['fleet_size'] = simulator.num_drivers
    final_record['total_time'] = (env_params['t_end'] - env_params['t_initial']) # s
    final_record['total_requests'] = len(final_record_df)
    final_record['speed'] = simulator.vehicle_speed
    matched = final_record_df[14] == 1
    final_record['matched_requests'] = len(final_record_df[matched])
    final_record['matching_rate'] = final_record['matched_requests']/final_record['total_time']
    final_record['matching_time'] = (final_record_df[matched][12]).mean() # s
    final_record['trip_time'] = final_record_df[matched][1].mean() # s
    final_record['pickup_time'] = final_record_df[matched][11].mean() # s
    final_record['effective_orders_total_waiting_time'] = final_record['pickup_time'] + final_record['matching_time']
    final_record['mean_waiting_orders'] = np.array(simulator.waiting_orders).mean()
    final_record['max_waiting_orders'] = np.array(simulator.waiting_orders).max()
    final_record['vacant_vehicles'] = np.array(simulator.vacant_vehicles).mean()
    save_file_name = 'orders_' + str(env_params['ave_order']) + '_drivers_' + str(env_params['num_drivers']) + '_record'
    pickle.dump(final_record, open(load_path + 'Results/' + save_file_name + '.pickle', 'wb'))
    print(final_record)