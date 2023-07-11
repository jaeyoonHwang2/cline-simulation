import constants
from tcp_base import TcpEventBased
import numpy as np
import math


class OrcaWorker:

    def __init__(self, params, use_normalizer=True):

        self.ack_count = 0
        self.loss_count = 0
        self.rtt_sum = 0
        self.min_rtt = 0
        self.srtt = 0
        self.timestamp = 0
        self.interval_cnt = 0
        self.throughput = 0
        self.loss_rate = 0
        self.avg_rtt = 0
        self.epoch = 0 ### renaming_needed
        self.epoch_ = 0 ### renaming_needed
        self.r = 0 ### renaming_needed
        self.ret = 0 ### renaming_needed
        self.action = [0, 0]

        ##### TCP_Env_Wrapper
        self.params = params
        self.avg_delay = 0.0
        self.avg_thr = 0.0
        self.del_ = 0.0
        self.max_bw = 0.0
        self.max_cwnd = 0.0
        self.max_smp = 0.0
        self.min_del = 9999999.0

        # var for obs
        self.z0 = np.zeros(15)
        self.target = 0
        self.pacing_rate = 0
        self.srtt_ms = 0
        self.retrans_out = 0
        self.max_packets_out = 0
        self.min_rtt = 0
        self.save = 0
        self.rxTime_last = 0

        s1_rec_buffer = np.zeros([s_dim])


        self.use_normalizer=use_normalizer
        if self.use_normalizer==True:
            self.normalizer=Normalizer(params)
        else:
            self.normalizer=None

    def collect_network_info(self, obs):
        self.ack_count += 1
        if not (obs[11]): self.loss_count += 1
        self.rtt_sum += obs[9] / 1000

        if not self.min_rtt:
            self.min_rtt = obs[9] / 1000
        else:
            self.min_rtt = min(self.min_rtt, obs[9] / 1000)

    def count_passed_interval(self, obs):
        if not self.timestamp: self.timestamp = int(obs[2] / constants.MTP) - 1
        self.interval_cnt = int(obs[2] / constants.MTP) - self.timestamp

    def calculate_network_state(self):
        total_received_bits = self.ack_count * constants.MTU_BYTES * 8
        total_lost_bits = self.loss_count * constants.MTU_BYTES * 8
        total_time_passed = self.interval_cnt * (constants.MTP / (1000 * 1000))
        self.throughput = total_received_bits / total_time_passed * constants.BPS_TO_MBPS
        self.loss_rate = total_lost_bits / total_time_passed * constants.BPS_TO_MBPS
        self.avg_rtt = self.rtt_sum / self.ack_count

    def get_action(self, obs):

        self.epoch += 1
        self.epoch_ += 1









    def reset(self, obs, mtpThr, mtpD, mtpL, srtt, dMin, numAcked):  ######## IS_DONE ########

        self.pre_samples = 0.0
        self.new_samples = 0.0
        self.avg_delay = 0.0
        self.avg_thr = 0.0
        self.del_ = 0.0
        self.max_bw = 0.0
        self.max_cwnd = 0.0
        self.max_smp = 0.0
        self.min_del = 9999999.0

        # var for obs
        self.z0 = np.zeros(15)
        self.target = 0
        self.pacing_rate = 0
        self.srtt_ms = 0
        self.retrans_out = 0
        self.max_packets_out = 0
        self.min_rtt = 0
        self.save = 0
        self.rxTime_last = 0
        self.pre_samples = 0.0
        self.new_samples = 0.0
        self.avg_delay = 0.0
        self.avg_thr = 0.0
        self.del_ = 0.0
        self.max_bw = 0.0
        self.max_cwnd = 0.0
        self.max_smp = 0.0
        self.min_del = 9999999.0

        self.obs_to_state(obs, mtpThr, mtpD, mtpL, srtt, dMin, numAcked)
        state, delay_, rew0, error_code = self.get_state()
        return state

    def get_dims_info(self):
        return self.params.dict['state_dim'], self.params.dict['action_dim']

    def get_action_info(self):
        action_scale = np.array([1.])
        action_range = (-action_scale, action_scale)
        return action_scale, action_range

    def obs_to_state(self, obs, mtpThr, mtpD, mtpL, srtt, dMin, numAcked):

        self.z0[0] = mtpD  # rtt_ms (orca-server-mahimahi.cc)
        self.z0[1] = mtpThr  # avg thr bps
        self.z0[2] = numAcked  # num of rtt samples used for averaging
        self.z0[3] = obs[2] / 1000000  # time after simulation started
        # self.z0[4] = 0 #### NEED NOT
        self.z0[5] = obs[5]  # cwnd_u
        self.z0[6] = obs[5] * 8 / (srtt / 1000)  # pacing_rate
        self.z0[7] = mtpL  # loss_rate
        self.z0[8] = srtt  # srtt_ms
        # self.z0[9] = obs[4] # snd_ssthresh
        # self.z0[10] = obs[8] # packets_out
        # self.z0[11] = 0 # retrans_out
        # self.z0[12] = 0 # max packets_out in last window
        self.z0[13] = obs[6]  # mss
        self.z0[14] = dMin  # min_rtt (d_min) # printed correctly?

        # print ("OBSERVATION ", self.z0)

    def get_state(self, evaluation=False):  # divide this into N; function per an observation

        s0 = self.z0
        reward = 0
        state = np.zeros(1)

        if len(s0) == (self.params.dict['input_dim']):
            d = s0[0]
            thr = s0[1]
            samples = s0[2]
            delta_t = s0[3]
            target_ = s0[4]
            cwnd = s0[5]
            pacing_rate = s0[6]
            loss_rate = s0[7]
            srtt_ms = s0[8]
            # snd_ssthresh=s0[9]
            # packets_out=s0[10]
            # retrans_out=s0[11]
            # max_packets_out=s0[12]
            mss = s0[13]
            min_rtt = s0[14]

            if self.use_normalizer == True:
                if evaluation != True:
                    self.normalizer.observe(s0)
                s0 = self.normalizer.normalize(s0)
                min_ = self.normalizer.stats()
            else:
                min_ = s0 - s0

            d_n = s0[0] - min_[0]
            thr_n = s0[1]
            thr_n_min = s0[1] - min_[1]
            samples_n = s0[2]
            samples_n_min = s0[2] - min_[2]
            delta_t_n = s0[3]
            delta_t_n_min = s0[3] - min_[3]

            cwnd_n_min = s0[5] - min_[5]
            pacing_rate_n_min = s0[6] - min_[6]
            loss_rate_n_min = s0[7] - min_[7]
            srtt_ms_min = s0[8] - min_[8]
            # snd_ssthresh_min=s0[9]-min_[9]
            # packets_out_min=s0[10]-min_[10]
            # retrans_out_min=s0[11]-min_[11]
            # max_packets_out_min=s0[12]-min_[12]
            # mss_min=mss-min_[13]
            min_rtt_min = min_rtt - min_[14]

            if self.use_normalizer == False:
                thr_n = thr_n
                thr_n_min = thr_n_min
                samples_n_min = samples_n_min
                cwnd_n_min = cwnd_n_min
                loss_rate_n_min = loss_rate_n_min
                d_n = d_n
            if self.max_bw < thr_n_min:
                self.max_bw = thr_n_min
            if self.max_cwnd < cwnd_n_min:
                self.max_cwnd = cwnd_n_min
            if self.max_smp < samples_n_min:
                self.max_smp = samples_n_min
            if self.min_del > d_n:
                self.min_del = d_n

            ################# Transfer all of the vars. to Rate/Max(Rate) space
            # cwnd_bytes= cwnd_n_min*mss_min
            # cwnd_n_min=(cwnd_bytes*1000)/srtt_ms_min
            # snd_ssthresh_min=(snd_ssthresh_min*mss_min*1000)/srtt_ms_min
            # packets_out_min=(packets_out_min*mss_min*1000)/srtt_ms_min
            # retrans_out_min=(retrans_out_min*mss_min*1000)/srtt_ms_min
            # max_packets_out_min=(max_packets_out_min*mss_min*1000)/srtt_ms_min
            # inflight_bytes=(packets_out-samples)*mss_min*1000

            if min_rtt_min * (self.params.dict['delay_margin_coef']) < srtt_ms_min:  # out of delay budget
                delay_metric = (min_rtt_min * (self.params.dict['delay_margin_coef'])) / srtt_ms_min
            else:  # within delay budget
                delay_metric = 1

            reward = (thr_n_min - 5 * loss_rate_n_min) / self.max_bw * delay_metric
            if self.max_bw != 0:
                state[0] = thr_n_min / self.max_bw  # factor1 (achieve highThr)
                tmp = pacing_rate_n_min / self.max_bw
                if tmp > 10:
                    tmp = 10
                state = np.append(state, [tmp])
                state = np.append(state, [5 * loss_rate_n_min / self.max_bw])  # factor2 (loss is critical)
                # print("MAX_BW IS ", self.max_bw)

            else:
                state[0] = 0
                state = np.append(state, [0])
                state = np.append(state, [0])
            state = np.append(state, [samples / cwnd])
            state = np.append(state, [delta_t_n])
            state = np.append(state, [min_rtt_min / srtt_ms_min])
            # print ("how is delay ", min_rtt_min, srtt_ms_min)
            state = np.append(state, [delay_metric])  # factor3 (achieve low delay)
            # THESE TWO STATES ARE FUCKING OVELAPPING. WHY THE FUCK WOULD ANYONE WANT TO SEND BORING COPY OF AN STATE WASTING ADDITIONAL 1/7 OF BW FUCKERS
            print(5, "STATE ", state)
            return state, d, reward, True
        else:
            return state, 0.0, reward, False

    def step(self, obs, mtpThr, mtpD, mtpL, srtt, dMin, numAcked, eval_=False):
        self.obs_to_state(obs, mtpThr, mtpD, mtpL, srtt, dMin, numAcked)
        s1, delay_, rew0, error_code = self.get_state(evaluation=eval_)
        return s1, rew0, False, error_code

##### TCP_Env_Wrapper
class Normalizer():
    def __init__(self, params):
        self.params = params
        self.n = 1e-5
        num_inputs = self.params.dict['input_dim']
        self.mean = np.zeros(num_inputs)
        self.mean_diff = np.zeros(num_inputs)
        self.var = np.zeros(num_inputs)
        self.dim = num_inputs
        self.min = np.zeros(num_inputs)

    def observe(self, x):
        self.n += 1
        last_mean = np.copy(self.mean)
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = self.mean_diff / self.n

    def normalize(self, inputs):
        obs_std = np.sqrt(self.var)
        a = np.zeros(self.dim)
        if self.n > 2:
            a = (inputs - self.mean) / obs_std
            for i in range(0, self.dim):
                if a[i] < self.min[i]:
                    self.min[i] = a[i]
            return a
        else:
            return np.zeros(self.dim)

    def normalize_delay(self, delay):
        obs_std = math.sqrt(self.var[0])
        if self.n > 2:
            return (delay - self.mean[0]) / obs_std
        else:
            return 0

    def stats(self):
        return self.min


