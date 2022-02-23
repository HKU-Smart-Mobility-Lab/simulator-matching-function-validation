from config import *
import pickle
import numpy as np
import pandas as pd

class Driver:
    def __init__(self, **kwargs):
        self.num_drivers = kwargs.pop('num_drivers', 100)
        self.df = pd.DataFrame({'driver_id': np.arange(self.num_drivers), 'start_time': 0, 'end_time':86400})
        self.side_length = kwargs.pop('grid_system_side_length', 10000)
        self.grid_length = kwargs.pop('grid_system_grid_length', 1000)
        self.num_grid = self.side_length // self.grid_length

    def create_coor(self):
        self.df[['lng', 'lat']] = np.random.rand(self.num_drivers, 2)*self.side_length
        self.df['direction'] = np.random.randint(0, 4, size=self.num_drivers)

    def get_grid_id(self):
        grid_x = self.df['lng'] // self.grid_length
        grid_y = self.df['lat'] // self.grid_length
        self.df['grid_id'] = grid_y * self.num_grid + grid_x

def create_driver():
    dr = Driver(**env_params)
    dr.create_coor()
    dr.get_grid_id()
    pickle.dump(dr.df, open('drivers.pickle', 'wb'))