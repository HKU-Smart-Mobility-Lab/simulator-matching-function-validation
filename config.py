env_params = {
't_initial' :0,
't_end' : 21600, # 21600
'delta_t' : 2,
'vehicle_speed' : 11.11,  # 40km/h, 11.11m/s
'driver_sample_ratio' : 1,
'order_sample_ratio' : 1,
'maximum_wait_time_mean' : 300,  # 5 min
'maximum_wait_time_std' : 0,
'request_interval' : 2,
'max_idle_time': 60,  # r1
'pickup_dis_threshold': 1000,  # unit: m
# 'cruising_prob_file_name' : 'cruising_prob_matrix_equally_dist',
'method' : 'pickup_distance',
'simulator_mode' : 'simulator_mode',
'request_file_name' : 'requests',
'driver_file_name' : 'drivers',
'experiment_date': 'test_day',
'grid_system_side_length' : 10000,  # unit: m
'grid_system_grid_length': 1000,  # unit: m
'ave_order' : 0.02,  # order number per delta_t
'num_drivers': 1000,
}
