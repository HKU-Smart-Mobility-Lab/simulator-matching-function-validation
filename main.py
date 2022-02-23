from simulator_env import Simulator
from config import *
from utilities import save_data
from Create_Drivers import create_driver
from Create_Records import create_records

if __name__ == "__main__":
    for i in [0.02, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]:
        for j in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
            num_order_per_delta_t = i
            env_params['time_period'] = int((env_params['t_end']) / env_params['delta_t'])
            num_driver = j * 100
            env_params['ave_order'] = num_order_per_delta_t
            env_params['num_drivers'] = num_driver

            create_driver()
            create_records()

            simulator = Simulator(**env_params)
            simulator.reset()

            for k in range(simulator.finish_run_step):
                simulator.step()
                if k %500 ==0:
                    print(simulator.current_step)
            save_data(simulator, env_params)
