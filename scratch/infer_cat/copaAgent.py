#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1. COPA with various velocity
2. No slow start
"""
import numpy as np
from collections import deque
from tcp_base import TcpEventBased
import random

class Sender_COPA(TcpEventBased):
    #def __init__(self):
    def __init__(self, segment_size=1, delta = 0.5, flow_ID = 0, initial_cwnd = 1 , mode = 'ss', v = 1, eq = 1, time = 1):
        super(Sender_COPA, self).__init__()
        # Parameters and network for learning
        self.segment_size = segment_size
        # For data monitoring
        self.g = 0.1
        self.RTT_list = deque([],maxlen = 2000)
        self.min_RTT = 100 # minimum RTT so far
        self.srtt = 0 # standing RTT
        self.delta = delta
        #For convenience
        self.current_cwnd_packets = initial_cwnd
        self.direction = 1
        self.RTT_standing = 0
        self.target = 0
        self.current_rate = 0
        self.mode = mode # ss for slow start, n for normal
        self.last_update = 0
        self.counter = 0
        # hwang var
        self.velocity = 1 # initial value
        self.directionUp = True 
        self.saveLastCwnd = 0 
        self.saveLastDirection = True
        self.numRtt = 0 
        

    def reset(self, time):
        # For data monitoring
        self.g = 0.1
        self.RTT_list = deque([], maxlen=2000)
        self.min_RTT = 100  # minimum RTT so far
        self.srtt = 0  # standing RTT
        self.delta = delta
        # For convenience
        self.current_cwnd_packets = self.initial_cwnd
        self.direction = 1
        self.RTT_standing = 0.002
        self.target = 0
        self.current_rate = 0
        self.mode = mode # ss for slow start, n for normal
        self.last_update = time
        self.counter = 0
        # hwang var
        self.velocity = 1 # initial value
        self.directionUp = True 
        self.saveLastCwnd = 0 
        self.saveLastDirection = True
        self.numRtt = 0 


    def get_action(self, obs):
        # RTT record
        if not(self.counter): self.srtt = obs[9] * 0.000001
        self.RTT_list.append([obs[2] * 0.000001, obs[9] * 0.000001])
        if self.min_RTT > obs[10]*0.000001: self.min_RTT = obs[10]*0.000001
        self.estimate_RTT_standing(obs)
        self.srtt = (1 - self.g) * self.srtt + self.g * (obs[9] * 0.000001)
        if self.RTT_standing > self.min_RTT: target = 1 / (self.delta * (self.RTT_standing - self.min_RTT))
        else: target = 10000000000

        if self.mode == 'ss': new_cwnd_packets = self.update_cwnd_ss(target, obs)
        else: new_cwnd_packets = self.update_cwnd(target, obs)
        new_cwnd_bytes = int(max(2, new_cwnd_packets)) * self.segment_size
        new_pacing_rate = int(2 * (new_cwnd_bytes / self.RTT_standing) * 8)
        action = [new_pacing_rate, new_cwnd_bytes]
        self.counter += 1
        return action

    def update_velocity(self, obs):

        #####
        if (self.current_cwnd_packets > self.saveLastCwnd): self.directionUp = True
        elif (self.current_cwnd_packets < self.saveLastCwnd): self.directionUp = False
        else: self.directionUp = ~(self.directionUp)

        if (self.directionUp == self.saveLastDirection):
            if (self.numRtt < 2): self.numRtt += 1
            else: self.velocity *= 2

        else: # direction changed
            self.numRtt = 0
            self.velocity = 1

        self.saveLastCwnd = self.current_cwnd_packets
        self.saveLastDirection = self.directionUp

        print (obs[0]-1, 'velocity', obs[2] / 1000000, self.velocity)

    def update_cwnd_ss(self, target, obs, RTT_standing):
        current_rate = (obs[5] / obs[6]) / RTT_standing
        # obs[5]/obs[6] : number of packets => current rate : number of packets per unit time

        self.current_rate = current_rate
        if current_rate > target:
            self.mode = 'n' # convert to the normal mode
            new_cwnd_packets = self.current_cwnd_packets
        else:
            if obs[2]*0.000001 > self.last_update + RTT_standing + self.min_RTT:
                self.last_update = obs[2] * 0.000001
                new_cwnd_packets = 2 * self.current_cwnd_packets
                self.current_cwnd_packets = new_cwnd_packets
            else:
                new_cwnd_packets = self.current_cwnd_packets
        return new_cwnd_packets

    def update_cwnd(self, target, obs):
        
        current_rate = (obs[5] / obs[6]) / self.RTT_standing
        self.current_rate = current_rate

        if current_rate < target:
            new_cwnd_packets = self.current_cwnd_packets + self.velocity/(self.delta*self.current_cwnd_packets)
        else:
            #new_cwnd_packets = max(int(self.current_cwnd_packets - self.velocity/(self.delta * self.current_cwnd_packets)), 2)
            new_cwnd_packets = max(self.current_cwnd_packets - self.velocity/(self.delta * self.current_cwnd_packets), 2)
        self.current_cwnd_packets = new_cwnd_packets
        
        return new_cwnd_packets

    def estimate_RTT_standing(self, obs):

        now = obs[2] * 0.000001
        tau = self.srtt / 2
        time = now - tau
        '''
        len_ = int(len(self.RTT_list))  # length of the memory
        idx = len_ - 1
        for i in range(len_):  # search from the most recent data
            if self.RTT_list[len_ - (1 + i)][0] <= time:  # [time, RTT]
                idx = i - 1  # i+1 elements
                break
        RTT = [list(self.RTT_list)[-(idx + 1):][i][1] for i in range(idx + 1)]
        '''
        len_ = int(len(self.RTT_list))
        idx = 0
        for i in range(len_):
            if self.RTT_list[i][0] > time:
                idx = i
                break

        self.standingRTT = self.RTT_list[idx][1]
        
        for i in range (len_-idx):
           
            if (self.RTT_list[idx+i][1] < self.standingRTT): self.standingRTT = self.RTT_list[idx+i][1]

