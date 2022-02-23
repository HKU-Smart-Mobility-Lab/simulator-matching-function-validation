from simulator_pattern import *
from numpy.random import choice

import math
from utilities import *

class Simulator:
    def __init__(self, **kwargs):
        # basic parameters: time & sample
        self.t_initial = kwargs['t_initial']
        self.t_end = kwargs['t_end']
        self.delta_t = kwargs['delta_t']
        self.vehicle_speed = kwargs['vehicle_speed']
        self.experiment_date = kwargs.pop('experiment_date', 'test_day')

        self.driver_sample_ratio = kwargs['driver_sample_ratio']
        self.order_sample_ratio = kwargs['order_sample_ratio']

        self.method = kwargs['method']
        self.pickup_dis_threshold = kwargs['pickup_dis_threshold']

        # wait cancel
        self.maximum_wait_time_mean = kwargs.pop('maximum_wait_time_mean', 120)
        self.maximum_wait_time_std = kwargs.pop('maximum_wait_time_std', 0)


        # pattern
        # experiment mode: train, test
        # experiment date: the date for experiment
        self.request_interval = kwargs.pop('request_interval', 60)
        self.simulator_mode = kwargs.pop('simulator_mode', 'simulator_mode')
        self.request_file_name = kwargs['request_file_name']
        self.driver_file_name = kwargs['driver_file_name']
        pattern_params = {'simulator_mode': self.simulator_mode, 'request_file_name':self.request_file_name, 'driver_file_name' : self.driver_file_name}
        pattern = SimulatorPattern(**pattern_params)


        # grid system initialization
        self.GS = GridSystem(**kwargs)
        self.GS.load_data()
        self.num_zone = self.GS.get_basics()

        # load cruising probability matrix r1
        # cruising_prob_file_name = kwargs['cruising_prob_file_name']
        # self.cruising_prob_mat = pickle.load(open(load_path + cruising_prob_file_name + '.pickle', 'rb'))
        self.max_idle_time = kwargs['max_idle_time']

        # get steps
        self.one_ep_steps = int((self.t_end - self.t_initial) // self.delta_t) + 1
        # self.pre_run_step = int((self.pre_run_time - self.t_initial) // self.delta_t)
        self.finish_run_step = int((self.t_end - self.t_initial) // self.delta_t)
        self.terminate_run_step = int((self.t_end - self.t_initial) // self.delta_t)

        # request tables
        # 暂时不考虑cancel prob，均设置为0
        self.request_columns = ['order_id', 'trip_time', 'origin_lng', 'origin_lat', 'dest_lng', 'dest_lat',
                                'immediate_reward', 'designed_reward', 'dest_grid_id', 't_start', 't_matched', 'pickup_time',
                                'wait_time', 't_end', 'status', 'driver_id', 'maximum_wait_time', 'cancel_prob',
                                'pickup_distance', 'weight']

        self.wait_requests = None
        self.matched_requests = None
        self.requests_final_records = None

        # driver tables
        self.driver_columns = ['driver_id', 'start_time', 'end_time', 'lng', 'lat', 'direction', 'grid_id', 'status',
                               'target_loc_lng', 'target_loc_lat', 'target_grid_id', 'remaining_time',
                               'matched_order_id', 'total_idle_time']
        self.num_drivers = kwargs.pop('num_drivers', 100)
        self.driver_table = None

        # order and driver databases
        self.sampled_driver_info = pattern.driver_info
        self.sampled_driver_info['grid_id'] = self.sampled_driver_info['grid_id'].values.astype(int)
        self.request_all = pattern.request_all

        #  vacant vehicles and waiting passengers (order)
        self.vacant_vehicles = []
        self.waiting_orders = []

    def initial_base_tables(self):
        # demand $ supply patterns
        self.request_databases = deepcopy(self.request_all[self.experiment_date])
        self.driver_table = sample_all_drivers(self.sampled_driver_info, self.t_initial, self.t_end,
                                               self.driver_sample_ratio)
        self.driver_table['target_grid_id'] = self.driver_table['target_grid_id']

        # order list update status
        self.wait_requests = pd.DataFrame(columns=self.request_columns)  # state: (wait 0, matched 1, finished 2)
        self.matched_requests = pd.DataFrame(columns=self.request_columns)  # state: (wait 0, matched 1, finished 2)
        self.requests_final_records = []

        # time/day
        self.time = deepcopy(self.t_initial)
        self.current_step = int((self.time - self.t_initial) // self.delta_t)

        # vehicle/driver table update status
        self.finished_driver_id = []


    def reset(self):
        """
        reset env
        :return:
        """
        # state_repo is a list of states of repositioned drivers
        self.initial_base_tables()


    def update_info_after_matching_multi_process(self, matched_pair_actual_indexes):
        new_matched_requests = pd.DataFrame([], columns=self.request_columns)
        update_wait_requests = pd.DataFrame([], columns=self.request_columns)

        matched_pair_index_df = pd.DataFrame(matched_pair_actual_indexes,
                                             columns=['order_id', 'driver_id', 'weight', 'pickup_distance'])
        matched_order_id_list = matched_pair_index_df['order_id'].values.tolist()
        con_matched = self.wait_requests['order_id'].isin(matched_order_id_list)
        con_keep_wait = self.wait_requests['wait_time'] <= self.wait_requests['maximum_wait_time']

        #when the order is matched
        df_matched = self.wait_requests[con_matched].reset_index(drop=True)
        if df_matched.shape[0] > 0:
            idle_driver_table = self.driver_table[self.driver_table['status'] == 0]
            order_array = df_matched['order_id'].values

            # multi process if necessary
            cor_order = []
            cor_driver = []
            for i in range(len(matched_pair_index_df)):
                cor_order.append(np.argwhere(order_array == matched_pair_index_df['order_id'][i])[0][0])
                cor_driver.append(idle_driver_table[idle_driver_table['driver_id'] == matched_pair_index_df['driver_id'][i]].index[0])
            cor_driver = np.array(cor_driver)
            df_matched = df_matched.iloc[cor_order, :]

            #decide whether cancelled
            # currently no cancellation
            cancel_prob = np.zeros(len(matched_pair_index_df))
            prob_array = np.random.rand(len(cancel_prob))
            con_remain = prob_array >= cancel_prob

            # order after cancelled
            update_wait_requests = df_matched[~con_remain]

            # driver after cancelled
            self.driver_table.loc[cor_driver[~con_remain], ['status', 'remaining_time', 'total_idle_time']] = 0

            # order not cancelled
            new_matched_requests = df_matched[con_remain]
            new_matched_requests['t_matched'] = self.time
            new_matched_requests['pickup_distance'] = matched_pair_index_df[con_remain]['pickup_distance'].values
            new_matched_requests['pickup_time'] = new_matched_requests['pickup_distance'].values / self.vehicle_speed
            new_matched_requests['t_end'] = self.time + new_matched_requests['pickup_time'].values + new_matched_requests['trip_time'].values
            new_matched_requests['status'] = 1
            new_matched_requests['driver_id'] = matched_pair_index_df[con_remain]['driver_id'].values

            # driver not cancelled
            self.driver_table.loc[cor_driver[con_remain], 'status']  = 1
            self.driver_table.loc[cor_driver[con_remain], 'target_loc_lng'] = new_matched_requests['dest_lng'].values
            self.driver_table.loc[cor_driver[con_remain], 'target_loc_lat'] = new_matched_requests['dest_lat'].values
            self.driver_table.loc[cor_driver[con_remain], 'target_grid_id'] = new_matched_requests['dest_grid_id'].values
            self.driver_table.loc[cor_driver[con_remain], 'remaining_time'] = new_matched_requests['t_end'].values \
                                                                              - new_matched_requests['t_matched'].values
            self.driver_table.loc[cor_driver[con_remain], 'matched_order_id'] = new_matched_requests['order_id'].values
            self.driver_table.loc[cor_driver[con_remain], 'total_idle_time'] = 0

            self.requests_final_records += new_matched_requests.values.tolist()

        # when the order is not matched
        update_wait_requests = pd.concat([update_wait_requests, self.wait_requests[~con_matched & con_keep_wait]],axis=0)
        self.requests_final_records += self.wait_requests[~con_matched & ~con_keep_wait].values.tolist()


        return new_matched_requests, update_wait_requests


    def step_bootstrap_new_orders(self):
       #generate new orders

       count_interval = int(math.floor(self.time / self.request_interval))
       self.request_database = self.request_databases[str(count_interval * self.request_interval)]
       weight_array = np.ones(len(self.request_database))

       if self.method == 'instant_reward':
           for i, request in enumerate(self.request_database):
               weight_array[i] = request[7]
       elif self.method == 'pickup_distance':
           pass

       np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
       order_id = [request[0] for request in self.request_database]
       requests = np.array([request[3:] for request in self.request_database])
       column_name = ['origin_lng', 'origin_lat', 'dest_lng', 'dest_lat', 'immediate_reward', 'trip_distance','trip_time', 'designed_reward', 'dest_grid_id']

       if len(requests) > 0:
           wait_info = pd.DataFrame(requests, columns=column_name)
           wait_info['dest_grid_id'] = wait_info['dest_grid_id'].values.astype(int)
           wait_info['order_id'] = order_id
           wait_info['t_start'] = self.time
           wait_info['wait_time'] = 0
           wait_info['status'] = 0
           wait_info['maximum_wait_time'] = self.maximum_wait_time_mean
           wait_info['cancel_prob'] = 0
           wait_info['weight'] = weight_array
           wait_info = wait_info.drop(columns=['trip_distance'])
           self.wait_requests = pd.concat([self.wait_requests, wait_info], ignore_index=True)

       return

    def cruising_decision(self, grid_id_indices):
        dec = []
        for g in grid_id_indices:
            a = np.where(self.GS.adj_mat[g] == 1)[0]
            b = random.randint(0, len(a) - 1)
            dec.append(a[b])
        return np.array(dec)



    def update_state(self):
        # update next state
        # update driver information
        self.driver_table['remaining_time'] = self.driver_table['remaining_time'].values - self.delta_t
        loc_negative_time = self.driver_table['remaining_time'] <= 0
        loc_idle = self.driver_table['status'] == 0
        loc_on_trip = self.driver_table['status'] == 1

        self.driver_table.loc[loc_negative_time, 'remaining_time'] = 0
        self.driver_table.loc[loc_negative_time, 'lng'] = self.driver_table.loc[
            loc_negative_time, 'target_loc_lng'].values
        self.driver_table.loc[loc_negative_time, 'lat'] = self.driver_table.loc[
            loc_negative_time, 'target_loc_lat'].values
        self.driver_table.loc[loc_negative_time, 'grid_id'] = self.driver_table.loc[
            loc_negative_time, 'target_grid_id'].values.astype(int)
        self.driver_table.loc[loc_idle, 'total_idle_time'] += self.delta_t
        self.driver_table.loc[loc_negative_time & loc_on_trip, 'status'] = 0

        # pickle.dump(self.driver_table, open(load_path + 'driver_table_1.pickle', 'wb'))
        # #cruising decision
        # for all drivers with no matched orders, cruise speed*delta t
        if len(self.driver_table.loc[loc_idle])>0:
            target = self.driver_table.loc[loc_idle, ['lng', 'lat']].values + (
                        self.driver_table.loc[loc_idle].apply(cruise, axis=1, result_type='expand') * self.vehicle_speed * self.delta_t).values
            valid_target = np.where(target > self.GS.side_length, self.GS.side_length , target)
            valid_target = np.where(valid_target < 0, 0, valid_target)

            grid_id = cal_grid_id(valid_target, self.GS.side_length, self.GS.grid_length).astype(int)

            self.driver_table.loc[loc_idle, ['lng', 'lat']] = valid_target
            self.driver_table.loc[loc_idle, 'grid_id'] = grid_id

            self.driver_table.loc[loc_idle, ['target_loc_lng', 'target_loc_lat']] = valid_target
            self.driver_table.loc[loc_idle, 'target_grid_id'] = grid_id

            # for those who have reached the max idle time, change the direction
            loc_long_idle = self.driver_table['total_idle_time'] == self.max_idle_time
            if len(self.driver_table.loc[loc_long_idle]) >0:
                self.driver_table.loc[loc_long_idle, 'direction'] = np.random.randint(0, 4, size=len(self.driver_table.loc[loc_long_idle]))
                self.driver_table.loc[loc_long_idle, 'total_idle_time'] = 0

        finished_driver_id = self.driver_table.loc[loc_negative_time & loc_on_trip, 'driver_id'].values.tolist()
        self.driver_table.loc[loc_negative_time & loc_on_trip, 'matched_order_id'] = 'None'

        # finished order deleted
        self.finished_driver_id.extend(finished_driver_id)
        if len(self.finished_driver_id) > 0:
            self.matched_requests = self.matched_requests[~self.matched_requests['driver_id'].isin(finished_driver_id)]
            self.matched_requests = self.matched_requests.reset_index(drop=True)
            self.finished_driver_id = []

        # wait list update
        self.wait_requests['wait_time'] += self.delta_t

        return

    def update_time(self):
        # time counter
        self.time += self.delta_t
        self.current_step = int((self.time - self.t_initial) // self.delta_t)
        return

    def step(self):
        # Step 1: bipartite matching
        wait_requests = deepcopy(self.wait_requests)
        driver_table = deepcopy(self.driver_table)
        matched_pair_actual_indexes = KM_simulation(wait_requests, driver_table, self.method, self.pickup_dis_threshold)

        # step 2
        # removed matched requests from wait list, and put them into matched list
        # delete orders with too much waiting time
        df_new_matched_requests, df_update_wait_requests = self.update_info_after_matching_multi_process(matched_pair_actual_indexes)
        self.matched_requests = pd.concat([self.matched_requests, df_new_matched_requests], axis=0)
        self.matched_requests = self.matched_requests.reset_index(drop=True)
        self.wait_requests = df_update_wait_requests.reset_index(drop=True)

        # Step 3: generate new orders
        self.step_bootstrap_new_orders()

        # Step 4: update next state
        self.update_state()
        self.update_time()

        # some records
        self.waiting_orders.append(len(self.wait_requests))
        self.vacant_vehicles.append(len(self.driver_table[self.driver_table['status'] == 0]))

        return

