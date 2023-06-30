import constants


class CubicAgent:

    def __init__(self):
        self.new_valid_timestamp = False
        self.no_first_loss = True
        self.max_cwnd = 10000000000
        self.epoch_start = 0   
        self.target = 0
        self.K = 0
        self.origin_point = 0
        self.cWnd_cnt = 0
        self.cnt = 0

        self.Uuid = 0
        self.simTime_us = 0
        self.observed_cwnd = 0
        self.segment_size = 0
        self.segments_acked = 0
        self.called_func = 0

        self.last_cwnd_decision = 0
        self.cwnd_decision = 0
        self.pacing_rate_decision = 0

        self.min_rtt_us = 0
        self.srtt_sec = 0
        self.mss_to_mtu = 0

    def get_action(self, obs, srtt_sec, min_rtt_us):
        self.min_rtt_us = min_rtt_us
        self.srtt_sec = srtt_sec

        self.monitor_obs(obs)

        if self.segments_acked: 
            if self.no_first_loss:
                self.slow_start()
            else:
                self.congestion_avoidance()
        else:  # no ack - initialize window (no reason)
            self.cwnd_decision = 2 * self.segment_size

        # if not (srtt): srtt = 120
        self.bound_cwnd_decision()
        self.calculate_pacing_rate()
        self.last_cwnd_decision = self.cwnd_decision
        self.printing_decisions()
        actions = [self.pacing_rate_decision, self.cwnd_decision]
              
        return actions
    
    def printing_decisions(self):
        print(self.Uuid, self.simTime_us * constants.US_TO_SEC, "cwnd", self.last_cwnd_decision)

    def slow_start(self):
        if self.called_func: 
            self.cwnd_exponential_increase()
        else:  # packet loss
            self.cwnd_reduction()

    def congestion_avoidance(self):
        if self.called_func: 
            self.cwnd_cubic_increase()
        else:  # packet loss
            self.cwnd_reduction()

    def calculate_pacing_rate(self):
        if not self.mss_to_mtu:
            self.mss_to_mtu = (self.segment_size + constants.HEADER_SIZE) / self.segment_size
        cwnd_to_pacing_rate = self.cwnd_decision * 8 / self.srtt_sec * self.mss_to_mtu

        if self.no_first_loss:
            self.pacing_rate_decision = int(2 * cwnd_to_pacing_rate)
        # TRY if (self.no_first_loss & self.called_func): => if (self.no_first_loss):
        else: 
            self.pacing_rate_decision = int(1.2 * cwnd_to_pacing_rate)

    def monitor_obs(self, obs):
        self.Uuid = obs[0] - 1
        self.simTime_us = obs[2]
        self.observed_cwnd = obs[5]
        self.segment_size = obs[6]
        self.segments_acked = obs[7]
        # function from Congestion Algorithm (CA) interface:
        #  GET_SS_THRESH = 0 (packet loss),
        #  INCREASE_WINDOW (packet acked),
        #  PKTS_ACKED (unused),
        #  CONGESTION_STATE_SET (unused),
        #  CWND_EVENT (unused),
        self.called_func = obs[11]

    def cwnd_reduction(self):
        if self.no_first_loss:
            self.no_first_loss = False
        self.max_cwnd = self.last_cwnd_decision
        self.cwnd_decision = int(max(self.last_cwnd_decision * constants.BETA, 2 * self.segment_size))
        self.epoch_start = 0

    def bound_cwnd_decision(self):
        if self.cwnd_decision < 2 * self.segment_size:
            self.cwnd_decision = 2 * self.segment_size

    def config_target_equation(self):
        self.epoch_start = self.simTime_us
        
        if self.observed_cwnd < self.max_cwnd:
            self.K = ((self.max_cwnd - self.observed_cwnd) / (self.segment_size * constants.C)) ** (1 / 3)
            self.origin_point = self.max_cwnd / self.segment_size

        else:    
            self.K = 0
            self.origin_point = self.observed_cwnd / self.segment_size

    def calculate_target_equation(self):
        t = self.simTime_us + self.min_rtt_us - self.epoch_start            
        self.target = self.origin_point + constants.C * (((t * constants.US_TO_SEC) - self.K)**3)

    def calculate_cnt_from_target(self):
        observed_packet_counts = self.observed_cwnd / self.segment_size
        if self.target > observed_packet_counts:
            self.cnt = observed_packet_counts / (self.target - (self.observed_cwnd / self.segment_size))
        else:
            self.cnt = 100 * observed_packet_counts

    def cwnd_exponential_increase(self):
        self.cwnd_decision = int(self.observed_cwnd + 2 * self.segment_size)
    
    def cwnd_cubic_increase(self):
        if self.epoch_start <= 0:
            self.config_target_equation()
        self.calculate_target_equation()
        self.calculate_cnt_from_target()
        
        if self.cWnd_cnt > self.cnt:
            self.cwnd_decision = self.observed_cwnd + self.segment_size
            self.cWnd_cnt = 0

        else: 
            self.cwnd_decision = self.observed_cwnd
            self.cWnd_cnt += 1
        