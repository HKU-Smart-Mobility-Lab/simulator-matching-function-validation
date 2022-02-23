from config import *
import random
import pickle
import numpy as np

class Records:
    def __init__(self, **kwargs):
        self.side_length = kwargs.pop('grid_system_side_length', 10000)
        self.grid_length = kwargs.pop('grid_system_grid_length', 1000)
        self.speed = kwargs.pop('vehicle_speed', 11.11)
        self.num_grid = self.side_length // self.grid_length
        self.average_order = kwargs.pop('ave_order', 20)
        self.time_period = kwargs.pop('time_period', 1440)
        self.request_interval = kwargs.pop('request_interval', 2)

    def get_coordinate(self):
        res = []
        res.append((random.random()) * self.side_length)
        res.append((random.random()) * self.side_length)
        res.append((random.random()) * self.side_length)
        res.append((random.random()) * self.side_length)
        return res

    def cal_distance(self, coor):
        x1, y1, x2, y2 = coor
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        manhattan_dis = dx + dy
        return manhattan_dis

    def cal_price(self, dis):
        if dis <= 2000:
            price = 6
        else:
            price = 6+(dis-2000)*1.5/1000
        return round(price, 1)

    def cal_time(self, dis):
        return dis/self.speed

    def get_dest_grid_id(self, coor):
        _, _, x, y = coor
        grid_x = x//self.grid_length
        grid_y = y//self.grid_length
        return int(grid_y*self.num_grid+grid_x)

    def create_one_record(self, id):
        rec = []
        rec.extend([id, 0, 0])
        coor = self.get_coordinate()
        rec.extend(coor)
        dis = self.cal_distance(coor) # unit: m
        price = self.cal_price((dis))
        rec.append(price)
        rec.append(dis)
        rec.append(self.cal_time(dis))
        rec.append(1)
        rec.append(self.get_dest_grid_id(coor))
        return rec

    def get_keys(self):
        self.keys = []
        for i in range(self.time_period):
            self.keys.append(str(i*self.request_interval))

    def get_num_record(self):
        self.get_keys()
        order_num = np.random.poisson(lam=self.average_order, size=self.time_period)

        self.num_record = dict(zip(self.keys, order_num))

    def create_records(self):
        i = 1000000
        self.get_num_record()
        self.rec = {}
        for k in self.num_record.keys():
            rec_list = []
            for j in range(self.num_record[k]):
                current_rec = self.create_one_record(str(i))
                rec_list.append(current_rec)
                i = i+1
            self.rec[k] = rec_list

def create_records():
    rc = Records(**env_params)
    rc.create_records()
    final_rc = {}
    final_rc['test_day'] = rc.rec
    pickle.dump(final_rc, open('requests.pickle', 'wb'))






