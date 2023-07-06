'''
MIT License
Copyright (c) Chen-Yu Yen - Soheil Abbasloo 2020
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from collections import deque
import copa_constant

class copa_agent:

    def __init__(self, delta = 1 / 2, initial_cwnd = 1):

        self.velocity = 1 
        self.cwnd = 1
        self.next_tx_seq = 0
        self.next_tx_time = 0        
        self.ack_count = 0 
        self.loss_count = 0
        self.min_rtt_us = 0
        self.min_rtt_ms = 0
        self.min_rtt_sec = 0
        self.srtt_us = 0
        self.srtt_ms = 0
        self.srtt_sec = 0
        self.timestamp = 0 
        self.interval_cnt = 0
        self.throughput_mbps = 0
        self.loss_rate_mbps = 0
        self.new_valid_timestamp = False

        self.an_rtt_after_last_velocity_calculation = False
        self.delta = delta

        self.max_bw_mbps = 1000
        self.rtt_list_us = deque([], maxlen=2000)
        self.standing_rtt_us = 0
        self.standing_rtt_ms = 0
        self.standing_rtt_sec = 0
        self.slow_start = True
        self.target_rate = 0
        self.current_rate = 0
        self.last_update = 0
        self.cwnd_pkts = initial_cwnd
        self.cwnd_bytes = initial_cwnd * copa_constant.SEGMENT_SIZE
        self.pacing_rate_mbps = 0        
        self.last_cwnd_pkts = 0       
        self.action = [] 
        self.direction_up = True
        self.last_direction = True
        self.consecutive_direction = 0        

    def collect_network_samples(self, obs):

        self.ack_count += 1
        if not (obs[11]): self.loss_count += 1

        if not (self.min_rtt_us): self.min_rtt_us = obs[9]
        else: self.min_rtt_us = min(self.min_rtt_us, obs[9])
        self.min_rtt_ms = self.min_rtt_us / 1000
        self.min_rtt_sec = self.min_rtt_us / 1000000

        if (self.srtt_us): self.srtt_us = (1 - copa_constant.G) * self.srtt_us + copa_constant.G * obs[9]  
        else: self.srtt_us = obs[9] 
        self.srtt_ms = self.srtt_us / 1000
        self.srtt_sec = self.srtt_us / 1000000


    def monitor_network(self, obs, Uuid):
        
        self.new_valid_timestamp = (obs[7] & (int(obs[2] / copa_constant.TIMESTAMP_UNIT_US) != self.timestamp))

        if (self.new_valid_timestamp):
        
            if not (self.timestamp): self.timestamp = int(obs[2] / copa_constant.TIMESTAMP_UNIT_US) - 1        
            self.interval_cnt = int(obs[2] / copa_constant.TIMESTAMP_UNIT_US) - self.timestamp

            self.throughput_mbps = min(self.ack_count * 1500 * 8 / (self.interval_cnt * (copa_constant.TIMESTAMP_UNIT_US / (1000 * 1000))) / (1024 * 1024), self.max_bw_mbps)
            self.loss_rate_mbps = self.loss_count * 1500 * 8 / (self.interval_cnt * (copa_constant.TIMESTAMP_UNIT_US / (1000 * 1000))) / (1024 * 1024) 

            for i in range(self.interval_cnt):
                
                print (Uuid, (self.timestamp + 1 + i) * copa_constant.BASE_RTT, "throughput", self.throughput_mbps)
                print (Uuid, (self.timestamp + 1 + i) * copa_constant.BASE_RTT, "loss_rate", self.loss_rate_mbps)
                print (Uuid, (self.timestamp + 1 + i) * copa_constant.BASE_RTT, "srtt", self.srtt_ms)

            self.ack_count = 0
            self.loss_count = 0

            self.timestamp = int(obs[2] / copa_constant.TIMESTAMP_UNIT_US) 

        else: 
            
            pass   

    def get_action(self, obs):

        self.estimate_rtt_standing(obs)
        self.set_target_rate()
        self.set_current_rate(obs)
        self.update_cwnd(obs)
        self.calculate_action()

        return self.action
        
    def estimate_rtt_standing(self, obs):

        self.rtt_list_us.append([obs[2], obs[9]])
        time = obs[2] - (self.srtt_us / 2)
        index = 0
        for i in range(len(self.rtt_list_us)):
            if self.rtt_list_us[i][0] > time:
                index = i
                break
        
        self.standing_rtt_us = 1000000

        for i in range (len(self.rtt_list_us) - index):

            self.standing_rtt_us = min(self.rtt_list_us[index + i][1], self.standing_rtt_us)

        self.standing_rtt_ms = self.standing_rtt_us / 1000
        self.standing_rtt_sec = self.standing_rtt_us / 1000000
        
    def set_target_rate(self):

        if (self.standing_rtt_us > self.min_rtt_us): self.target_rate = 1 / (self.delta * (self.standing_rtt_sec - self.min_rtt_sec))
        else: self.target_rate = 10000000000        

    def set_current_rate(self, obs):

        self.current_rate = (obs[5] / obs[6]) / self.standing_rtt_sec
    
    def update_cwnd(self, obs):
        
        if (self.slow_start): self.slow_start_mode(obs)
        else: self.normal_mode()     

    def slow_start_mode(self, obs):
        
        if (self.current_rate > self.target_rate): self.slow_start = False            
        else: 
            if obs[2] > self.last_update + self.standing_rtt_us + self.min_rtt_us:
                self.last_update = obs[2]
                self.cwnd_pkts *= 2                

    def normal_mode(self):
        
        if (self.current_rate < self.target_rate): self.cwnd_pkts += self.velocity / (self.delta * self.cwnd_pkts)
        else: self.cwnd_pkts = max(self.cwnd_pkts - self.velocity / (self.delta * self.cwnd_pkts), 2)

    def calculate_action(self):
        
        self.cwnd_bytes = int(max(2, self.cwnd_pkts)) * copa_constant.SEGMENT_SIZE
        self.pacing_rate_mbps = int(2 * (self.cwnd_bytes / self.standing_rtt_sec) * 8)
        self.action = [self.pacing_rate_mbps, self.cwnd_bytes]

    def velocity_calculate(self, obs, Uuid):

        self.an_rtt_after_last_velocity_calculation = (self.next_tx_time <= obs[2] - obs[9])

        if (self.an_rtt_after_last_velocity_calculation):

            if (self.cwnd_pkts > self.last_cwnd_pkts): self.direction_up = True
            elif (self.cwnd_pkts < self.last_cwnd_pkts): self.direction_up = False
            else: self.direction_up = ~(self.direction_up)

            if (self.direction_up == self.last_direction):
                if (self.consecutive_direction < 2): self.consecutive_direction += 1
                else: self.velocity *= 2

            else:
                self.consecutive_direction = 0
                self.velocity = 1

            self.last_cwnd_pkts = self.cwnd_pkts
            self.last_direction = self.direction_up

            self.next_tx_seq = obs[14]
            self.next_tx_time = obs[2]

        
