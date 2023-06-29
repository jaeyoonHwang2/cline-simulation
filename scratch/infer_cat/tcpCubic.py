from tcp_base import TcpEventBased

class TcpCubic(TcpEventBased):
    """docstring for TcpCubic"""
    def __init__(self):
        super(TcpCubic, self).__init__()
        self.C = 0.4 # scaling constant
        self.beta = 717 / 1024 # multiplicative decrease factor
        self.cnt = 0
        self.cWnd_cnt = 0
        self.epoch_start = 0
        self.origin_point = 0
        self.K = 0
        self.T = 0
        self.max_cWnd = 1000000000 
        self.target = 0
        self.slow_start = True
        self.last_cubic_cwnd = 90
        self.rtt_loss_signal = False
        self.loss_available = True
        self.last_loss_time = 0
        self.loss_event = False
        self.cubic_cwnd = 90
        self.cubic_pacing_rate = 90

    def reset(self):
        self.C = 0.4 # scaling constant
        self.beta = 717 / 1024 # multiplicative decrease factor
        self.cnt = 0
        self.cWnd_cnt = 0
        self.epoch_start = 0
        self.origin_point = 0
        self.K = 0
        self.T = 0
        self.max_cWnd = 1000000000 
        self.target = 0
        self.slow_start = True
        self.last_cubic_cwnd = 90
        self.rtt_loss_signal = False
        self.loss_available = True
        self.last_loss_time = 0
        self.loss_event = False
        self.cubic_cwnd = 90
        self.loss_event = False

    def get_action(self, obs, srtt, dMin, done, info):
        # unique socket ID
        socketUuid = obs[0]
        # TCP env type: event-based = 0 / time-based = 1
        envType = obs[1]
        # sim time in us
        simTime_us = obs[2]
        # unique node ID
        nodeId = obs[3]
        # current ssThreshold
        ssThresh = obs[4]
        # current contention window size
        cWnd = obs[5]
        # segment size
        segmentSize = obs[6]
        # number of acked segments
        segments_acked = obs[7]
        # estimated bytes in flight
        bytesInFlight  = obs[8]
        # last estimation of RTT
        lastRtt_us  = obs[9]
        # min value of RTT
        minRtt_us  = obs[10]
        # function from Congestion Algorithm (CA) interface:
        #  GET_SS_THRESH = 0 (packet loss),
        #  INCREASE_WINDOW (packet acked),
        #  PKTS_ACKED (unused),
        #  CONGESTION_STATE_SET (unused),
        #  CWND_EVENT (unused),
        calledFunc = obs[11]
        # Congetsion Algorithm (CA) state:
        # 0 CA_OPEN = 0,
        # 1 CA_DISORDER,
        # 2 CA_CWR,
        # 3 CA_RECOVERY,
        # 4 CA_LOSS,
        # 5 CA_LAST_STATE
        caState = obs[12]
        # Congetsion Algorithm (CA) event:
        # 1 CA_EVENT_TX_START = 0,
        # 2 CA_EVENT_CWND_RESTART,
        # 3 CA_EVENT_COMPLETE_CWR,
        # 4 CA_EVENT_LOSS,
        # 5 CA_EVENT_ECN_NO_CE,
        # 6 CA_EVENT_ECN_IS_CE,
        # 7 CA_EVENT_DELAYED_ACK,
        # 8 CA_EVENT_NON_DELAYED_ACK,
        caEvent = obs[13]

        mss_to_mtu = (obs[6] + 60) / obs[6]
        
        buffer_bdp = 1
        self.rtt_loss_signal = obs[9] > 20000 * (1 + buffer_bdp)
        self.loss_available = self.last_loss_time + 20000 < simTime_us
        self.loss_event = calledFunc == 0

        if (segments_acked):

            if (self.rtt_loss_signal & self.loss_available):
                
                self.loss_event = True
                self.last_loss_time = simTime_us

            if (self.loss_event): 

                self.epoch_start = 0
                self.max_cWnd = self.last_cubic_cwnd
                self.cubic_cwnd = int(max(self.last_cubic_cwnd * self.beta, 2 * segmentSize))
                self.slow_start = False

            else:                

                if (self.slow_start): # slow start phase
                                                    
                    self.cubic_cwnd = int(cWnd + segmentSize)                

                else: # congestion avoidance phase

                    if (self.epoch_start <= 0): 

                        self.epoch_start = simTime_us

                        if (cWnd < self.max_cWnd): 
                            self.K = ((self.max_cWnd - cWnd) / (segmentSize * self.C))**(1/3)  
                            self.origin_point = self.max_cWnd / segmentSize

                        else: 
                            self.K = 0
                            self.origin_point = cWnd / segmentSize
                    
                    t = simTime_us + dMin - self.epoch_start            
                    self.target = self.origin_point + self.C * (((t/1000000)-self.K)**3)

                    if (self.target > (cWnd / segmentSize)): self.cnt = (cWnd / segmentSize) / (self.target-(cWnd / segmentSize))
                    else: self.cnt = 100 * cWnd / segmentSize
                    
                    if (self.cWnd_cnt > self.cnt): 
                        
                        self.cubic_cwnd = cWnd + segmentSize
                        self.cWnd_cnt = 0

                    else: 

                        self.cubic_cwnd = cWnd
                        self.cWnd_cnt += 1

        # clamp maximum cwnd 
        if (self.cubic_cwnd < 180): self.cubic_cwnd = 180
        if (self.slow_start): self.cubic_pacing_rate = int(2 * self.cubic_cwnd * 8 / (srtt / 1000) * mss_to_mtu)
        else: self.cubic_pacing_rate = int(1.2 * self.cubic_cwnd * 8 / (srtt / 1000) * mss_to_mtu)

        self.last_cubic_cwnd = self.cubic_cwnd
        self.last_cubic_pacing_rate = self.cubic_pacing_rate
        
        actions = [self.cubic_pacing_rate, self.cubic_cwnd]      
 
        return actions, self.slow_start
