#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 17:57:57 2020

@author: imran
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('Ads_CTR_Optimisation.csv')

import math

N=10000
d=10
ads_selected = []
number_of_selections = [0] * d
sums_of_reward = [0] * d
total_reward = 0
for n in range(0,N):
    max_upper_bound=0
    ad=0
    for i in range(0,d):
        if(number_of_selections[i] > 0):
            avg_reward = sums_of_reward[i]/number_of_selections[i]
            delta_i = math.sqrt(1.5 * math.log(n+1)/number_of_selections[i])
            upper_bound = avg_reward + delta_i
            
        else:
            upper_bound=1e400
        
        if max_upper_bound < upper_bound:
            max_upper_bound = upper_bound
            ad = i
    
    ads_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] + 1
    reward = df.values[n, ad]
    sums_of_reward[ad] = sums_of_reward[ad] + reward
    total_reward = total_reward + reward
    
plt.hist(ads_selected)
plt.title('Upper Confidence Bound')
plt.xlabel('ads versions')
plt.ylabel('Number of times selected')
plt.draw()