# !/usr/bin/python
# -*- coding:utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib.colors
from datetime import datetime


def draw_scatter(data):

    new_data_0 = data[data[:,2] == 0]
    new_data_1 = data[data[:,2] == 1]
    new_data_2 = data[data[:,2] == 2]
    new_data_3 = data[data[:,2] == 3]


    plt.scatter(new_data_0[:,0], new_data_0[:,1], color = 'red')
    plt.scatter(new_data_1[:,0], new_data_1[:,1], color = 'blue')
    plt.scatter(new_data_2[:,0], new_data_2[:,1], color = 'green')
    plt.scatter(new_data_3[:,0], new_data_3[:,1], color = 'yellow')   

def calc_distance(vec1, vec2):
    return math.sqrt(((vec1 - vec2)**2).sum())

def draw_canopy(data, cluster, center, far):

    colors = [
        'brown', 'green', 'blue', 'y', 'r', 'tan', 'dodgerblue', 'deeppink',
        'orangered', 'peru', 'blue', 'y', 'r', 'gold', 'dimgray', 'darkorange',
        'peru', 'blue', 'y', 'r', 'cyan', 'tan', 'orchid', 'peru', 'blue', 'y',
        'r', 'sienna'
    ]
    markers = [
        '*', 'h', 'H', '+', 'o', '1', '2', '3', ',', 'v', 'H', '+', '1', '2',
        '^', '<', '>', '.', '4', 'H', '+', '1', '2', 's', 'p', 'x', 'D', 'd',
        '|', '_'
    ]

    n = len(data)

    #draw center
    centers = len(center)

    #for all sample
    for idx in range(n):

        center_num = len(cluster[idx])

        values = []
        cols = []
        for center_idx in range(center_num):
            values.append( 100 / center_num)
            cols.append(colors[center.index(cluster[idx][center_idx])])

        
        plt.pie(values,colors=cols, center= (data[idx, 0], data[idx, 1]), radius= far * len(center) * 1.5/ len(cluster))

    plt.axis('equal')



def calc_candidate(cluster):

    cnt = len(cluster)

    for i in range(cnt):
        if len(cluster[i]) > 0:
            cnt = cnt - 1

    return cnt

def get_new_canopy(cluster, ran_idx):

    cnt = len(cluster)

    for i in range(cnt):

        if len(cluster[i]) == 0:
            if ran_idx > 1:
                ran_idx = ran_idx - 1
            else:
                return i

    return -1 
    
def generate_canopy(data, t_near, t_far):

    cnt = len(data)
    cluster = [[] for i in range(cnt)]
    center = []

    candidate_sample = cnt

    while candidate_sample > 0:

        ran_idx = np.random.randint(candidate_sample)
        data_idx = get_new_canopy(cluster, ran_idx)
        center.append(data_idx)

        if data_idx == -1:
            break
        
        for j in range(cnt):
            distance = calc_distance(data[data_idx], data[j])

            if distance < t_near:
                cluster[j] = [data_idx]

            elif distance < t_far:
                cluster[j].append(data_idx)

        candidate_sample = calc_candidate(cluster)

    return cluster, center


def main():

    #generate the sample data
    N = 100
    centers = 4
    data, y = ds.make_blobs(N, n_features=2, centers=centers,  random_state=2)

    new_data = (np.concatenate((data, np.reshape(y, (N,1))), axis = 1))

    #draw the original
    plt.subplot(121)
    draw_scatter(new_data)

    x_min = np.min(data[:,0])
    x_max = np.max(data[:,0])
    y_min = np.min(data[:,1])
    y_max = np.max(data[:,1])


    t_near = np.max([np.abs(x_max-x_min), np.abs(y_max - y_min)]) / (2 * centers)
    t_far = np.max([np.abs(x_max-x_min), np.abs(y_max - y_min)]) / centers


    cluster , center = generate_canopy(data, t_near, t_far)

    plt.subplot(122)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)


    my_x_ticks = np.arange(x_min, x_max, 0.5)
    my_y_ticks = np.arange(y_min, y_max, 0.5)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)

    draw_canopy(data, cluster, center, t_far)

    plt.show()


if __name__ == '__main__':
    t_start = datetime.now()
    main()
    t_end = datetime.now()
    usedtime = t_start - t_end
    print('[%s]' % usedtime)
