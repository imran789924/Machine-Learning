#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 00:55:31 2020

@author: imran
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('Ads_CTR_Optimisation.csv')

import random

N=10000
d=10
ads_selected = []
number_of_rewards_one = [0] * d
number_of_rewards_zero = [0] * d
total_reward = 0
for n in range(0,N):
    max_random=0
    ad=0
    for i in range(0,d):
        random_beta = random.betavariate(number_of_rewards_one[i]+1, number_of_rewards_zero[i]+1)
        
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    
    ads_selected.append(ad)
    reward = df.values[n, ad]
    if reward == 1:
        number_of_rewards_one[ad] = number_of_rewards_one[ad] + 1
    else:
        number_of_rewards_zero[ad] = number_of_rewards_zero[ad] + 1
    #print(max_upper_bound)
    total_reward = total_reward + reward
    
plt.hist(ads_selected)
plt.title('Ads selection Thomas sampling')
plt.xlabel('ads versions')
plt.ylabel('Number of times selected')
plt.draw()