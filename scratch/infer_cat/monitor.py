import constants


class MonitoringAgent:

    def __init__(self, cc_type):

        self.ack_count = 0
        self.loss_count = 0
        self.min_rtt_us = 0
        self.min_rtt_ms = 0
        self.min_rtt_sec = 0
        self.srtt_us = 0
        self.srtt_ms = 0
        self.srtt_sec = 0
        self.new_monitoring_period = False
        self.timestamp = 0
        self.interval_cnt = 0
        self.throughput_mbps = 0
        self.loss_rate_mbps = 0

        self.Uuid = 0
        self.sim_time_us = 0
        self.ssThresh = 0
        self.cwnd = 0
        self.segment_size = 0
        self.segments_acked = 0 
        self.bytes_in_flight = 0
        self.lastRtt_us = 0
        self.minRtt_us = 0
        self.called_func = 0
        self.caState = 0
        self.caEvent = 0

        self.cc_type = cc_type

    def monitor_network(self, obs):

        self.monitor_obs(obs)
        self.count_ack()
        self.count_loss()
        self.measure_min_rtt()
        self.calculate_srtt()
        self.change_rtt_timescale()
        self.is_it_new_monitoring_period()
        
        if self.new_monitoring_period:
            self.measure_network_performance()
        else:
            pass

        return self.pass_monitored_value()

    def pass_monitored_value(self):

        if self.cc_type == "cubic":
            return self.srtt_sec, self.min_rtt_us
        elif self.cc_type == "newreno":
            return

    def change_rtt_timescale(self):

        self.min_rtt_ms = self.min_rtt_us * constants.US_TO_MS
        self.min_rtt_sec = self.min_rtt_us * constants.US_TO_SEC
        self.srtt_ms = self.srtt_us * constants.US_TO_MS
        self.srtt_sec = self.srtt_us * constants.US_TO_SEC

    def monitor_obs(self, obs):
        self.Uuid = obs[0] - 1
        # sim time in us
        self.sim_time_us = obs[2]
        # current ssThreshold
        self.ssThresh = obs[4]
        # current contention window size
        self.cwnd = obs[5]
        # segment size
        self.segment_size = obs[6]
        # number of acked segments
        self.segments_acked = obs[7]
        # estimated bytes in flight
        self.bytes_in_flight = obs[8]
        # last estimation of RTT
        self.lastRtt_us = obs[9]
        # min value of RTT
        self.minRtt_us = obs[10]
        # function from Congestion Algorithm (CA) interface:
        #  GET_SS_THRESH = 0 (packet loss),
        #  INCREASE_WINDOW (packet acked),
        #  PKTS_ACKED (unused),
        #  CONGESTION_STATE_SET (unused),
        #  CWND_EVENT (unused),
        self.called_func = obs[11]
        # Congetsion Algorithm (CA) state:
        # 0 CA_OPEN = 0,
        # 1 CA_DISORDER,
        # 2 CA_CWR,
        # 3 CA_RECOVERY,
        # 4 CA_LOSS,
        # 5 CA_LAST_STATE
        self.caState = obs[12]
        # Congetsion Algorithm (CA) event:
        # 1 CA_EVENT_TX_START = 0,
        # 2 CA_EVENT_CWND_RESTART,
        # 3 CA_EVENT_COMPLETE_CWR,
        # 4 CA_EVENT_LOSS,
        # 5 CA_EVENT_ECN_NO_CE,
        # 6 CA_EVENT_ECN_IS_CE,
        # 7 CA_EVENT_DELAYED_ACK,
        # 8 CA_EVENT_NON_DELAYED_ACK,
        self.caEvent = obs[13]

    def count_ack(self):
        self.ack_count += 1

    def count_loss(self):
        if not self.called_func:
            self.loss_count += 1

    def measure_min_rtt(self):
        if not self.min_rtt_us:
            self.min_rtt_us = self.lastRtt_us
        else:
            self.min_rtt_us = min(self.min_rtt_us, self.lastRtt_us)

    def calculate_srtt(self):   
        if self.srtt_us:
            self.srtt_us = (1 - constants.G) * self.srtt_us + constants.G * self.lastRtt_us
        else:
            self.srtt_us = self.lastRtt_us

    def is_it_new_monitoring_period(self):
        period_end = int(self.sim_time_us / constants.TIMESTAMP_UNIT_US) != self.timestamp
        self.new_monitoring_period = (self.segments_acked & period_end)

    def measure_network_performance(self):
        if not self.timestamp:
            self.timestamp = int(self.sim_time_us / constants.TIMESTAMP_UNIT_US) - 1
        self.interval_cnt = int(self.sim_time_us / constants.TIMESTAMP_UNIT_US) - self.timestamp

        period_length = self.interval_cnt * (constants.TIMESTAMP_UNIT_US * constants.US_TO_SEC)
        packets_to_rate = constants.MTU_BYTES * constants.BYTE_TO_BITS / period_length * constants.BPS_TO_MBPS

        self.throughput_mbps = self.ack_count * packets_to_rate
        self.loss_rate_mbps = self.loss_count * packets_to_rate

        self.print_network_performance()

        self.ack_count = 0
        self.loss_count = 0

        self.timestamp = int(self.sim_time_us / constants.TIMESTAMP_UNIT_US)

    def print_network_performance(self):
        timestamp_unit_sec = constants.TIMESTAMP_UNIT_US * constants.US_TO_SEC

        for i in range(self.interval_cnt):
            print(self.Uuid, (self.timestamp + 1 + i) * timestamp_unit_sec, "throughput", self.throughput_mbps)
            print(self.Uuid, (self.timestamp + 1 + i) * timestamp_unit_sec, "loss_rate", self.loss_rate_mbps)
            print(self.Uuid, (self.timestamp + 1 + i) * timestamp_unit_sec, "srtt", self.srtt_ms)
            print(self.Uuid, (self.timestamp + 1 + i) * timestamp_unit_sec, "rtt", self.lastRtt_us)
