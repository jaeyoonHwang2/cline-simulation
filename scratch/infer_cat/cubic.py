import constants


class CubicAgent:

    def __init__(self):

        self.new_valid_timestamp = False
        self.slow_start_phase = True
        self.max_cwnd = 0
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
        self.first_loss = True

    def get_action(self, obs, srtt_sec, min_rtt_us):
        self.min_rtt_us = min_rtt_us
        self.srtt_sec = srtt_sec

        self.monitor_obs(obs)

        if not self.called_func: # packet loss       
            self.slow_start_phase = False

        if self.segments_acked:
            if self.slow_start_phase:
                self.slow_start()
            else:
                self.congestion_avoidance()
        else:  # no ack - initialize window (no reason)
            self.cwnd_decision = self.last_cwnd_decision

        self.last_cwnd_decision = self.cwnd_decision
        self.print_decisions()
        actions = [self.pacing_rate_decision, self.cwnd_decision]

        return actions

    def print_decisions(self):
        # print(self.Uuid, self.simTime_us * constants.US_TO_SEC, "cwnd", self.last_cwnd_decision)
        pass

    def slow_start(self):
        self.cwnd_increase_ss()
        self.calculate_pacing_rate_ss()

    def congestion_avoidance(self):
        if self.called_func:
            self.cwnd_increase_ca()
        else:  # packet loss
            self.cwnd_reduction()
        self.calculate_pacing_rate_ca()

    def calculate_pacing_rate_ss(self):
        if not self.mss_to_mtu:
            self.mss_to_mtu = (self.segment_size + constants.HEADER_SIZE) / self.segment_size
        cwnd_to_pacing_rate = self.cwnd_decision * 8 / self.srtt_sec * self.mss_to_mtu
        self.pacing_rate_decision = int(2 * cwnd_to_pacing_rate)

    def calculate_pacing_rate_ca(self):
        if not self.mss_to_mtu:
            self.mss_to_mtu = (self.segment_size + constants.HEADER_SIZE) / self.segment_size
        cwnd_to_pacing_rate = self.cwnd_decision * 8 / self.srtt_sec * self.mss_to_mtu
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
        self.cwnd_decision = int(max(self.last_cwnd_decision * constants.BETA, 2 * self.segment_size))
        if constants.FAST_CONV:
            self.max_cwnd = self.last_cwnd_decision * (2 - constants.BETA) / 2
        else:
            self.max_cwnd = self.last_cwnd_decision
        self.epoch_start = 0

    def config_target_equation(self):
        self.epoch_start = self.simTime_us

        if self.last_cwnd_decision <= self.max_cwnd:
            self.K = ((self.max_cwnd * (1 - constants.BETA)) / (self.segment_size * constants.C)) ** (1 / 3)
            self.origin_point = self.max_cwnd / self.segment_size
        else:
            self.K = 0
            self.origin_point = self.observed_cwnd / self.segment_size

    def calculate_target_equation(self):
        t = (self.simTime_us + self.min_rtt_us - self.epoch_start) * constants.US_TO_SEC
        self.target = self.origin_point + constants.C * ((t - self.K)**3)

    def calculate_cnt_from_target(self):
        observed_packet_counts = self.last_cwnd_decision / self.segment_size
        if self.target > observed_packet_counts:
            self.cnt = observed_packet_counts / (self.target - (self.last_cwnd_decision / self.segment_size))
        else:
            self.cnt = 100 * observed_packet_counts

    def cwnd_increase_ss(self):
        self.cwnd_decision = int(self.last_cwnd_decision + self.segment_size)

    def cwnd_increase_ca(self):
        if self.epoch_start <= 0:
            self.config_target_equation()
        self.calculate_target_equation()
        self.calculate_cnt_from_target()
        self.calculate_cwnd_from_cnt()

    def calculate_cwnd_from_cnt(self):
        if self.cWnd_cnt > self.cnt:
            self.cwnd_decision = self.last_cwnd_decision + self.segment_size
            self.cWnd_cnt = 0
        else:
            self.cwnd_decision = self.last_cwnd_decision
            self.cWnd_cnt += 1
        