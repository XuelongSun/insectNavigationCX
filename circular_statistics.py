# @File: circular_statistics.py
# @Info: useful functions to analysis the heading data of the homing agent
# @Author: Xuelong Sun, UoL, UK
# @Time: 2020-02-17

import numpy as np


def circular_statistics(d, acc=100):
    # d - the data
    # acc - the accurancy for slicing 2*pi
    d_ref = np.linspace(0, 2*np.pi, acc)
    n = len(d)
    num = np.zeros([acc])

    x = np.sum(np.cos(d))/n
    y = np.sum(np.sin(d))/n
    r = np.sqrt(x**2 + y**2)
    # calculate the mean
    mean = np.arctan2(y, x)
    # calculate the standard de
    sd0 = np.sqrt(-2*np.log(r))
    ci95 = 1.96 * sd0 / np.sqrt(n)
    for j in range(n):
        diff = abs(d_ref-d[j] % (2*np.pi))
        num[np.argmin(diff)] += 1

    return r, mean, sd0, ci95, num

def get_check_points_index_from_p(dis, pos_array, p):
    cp_ind = np.zeros(len(pos_array), np.int)
    for i in range(len(pos_array)):
        _dis = np.sqrt((pos_array[i, :] - p)[:,0]**2 + (pos_array[i,:] - p)[:,1]**2)
        if len(np.where(_dis <= dis)[0]) !=0:
            cp_ind[i] = np.where(_dis <= dis)[0][0]
        else:
            cp_ind[i] = 0
    return cp_ind

def get_check_points_index(dis, pos_array):
    check_dis = dis
    cp_ind = np.zeros(len(pos_array), np.int)
    for i in range(len(pos_array)):
        dis = np.sqrt((pos_array[i, :] - pos_array[i,0,:])[:,0]**2 + (pos_array[i,:] - pos_array[i,1,:])[:,1]**2)
        if len(np.where(dis >= check_dis)[0]) !=0:
            cp_ind[i] = np.where(dis >= check_dis)[0][0]
        else:
            cp_ind[i] = 0
    return cp_ind


def get_check_points_h(dis, pos_array, h):
    ck_ind = get_check_points_index(dis, pos_array)
    ck_ind = list(filter(lambda x: x!=0, ck_ind))
    ck_h = ([h[i][ck_ind[i]] for i in range(len(ck_ind))])
    return ck_h


def calculate_rf_motor(rf_mem, current_h, current_zm_p):
    return (current_h-(rf_mem - current_zm_p) + np.pi)%(np.pi*2) - np.pi

