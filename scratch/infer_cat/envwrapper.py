import numpy as np
import math
from utils import logger
import os
import sys
import constants
import signal
from tcp_base import TcpEventBased


class TcpEnvWrapper(TcpEventBased):

    def __init__(self, s_dim, params, use_normalizer=True):
        super(TcpEnvWrapper, self).__init__()
        self.params = params

        self.max_bw = 0.0
        self.max_cwnd = 0.0
        self.max_smp = 0.0
        self.min_del = 9999999.0

        self.obs = np.zeros(15)
        self.z0 = np.zeros(15)

        signal.signal(signal.SIGINT, self.handler_term)
        signal.signal(signal.SIGTERM, self.handler_term)
        self.use_normalizer = use_normalizer
        if self.use_normalizer:
            self.normalizer = Normalizer(params)
        else:
            self.normalizer = None
        if self.use_normalizer:
            _ = self.normalizer.load_stats()

        # updating part
        self.alpha = 0
        self.last_agent_decision = 0
        self.ack_count = 0
        self.loss_count = 0
        self.rtt_sum_ms = 0
        self.min_rtt_ms = 0
        self.srtt_ms = 0
        self.timestamp = 0
        self.interval_cnt = 0
        self.throughput = 0
        self.loss_rate = 0
        self.avg_rtt_ms = 0
        self.epoch_for_whole = 0
        self.epoch_for_this_epi = 0  # need renaming
        self.reward = 0
        self.ret = 0
        self.Uuid = 0
        self.terminal = False
        self.error_code = True

        self.new_mtp = True

        self.s0 = np.zeros(s_dim)
        self.s1 = np.zeros(s_dim)
        self.s0_rec_buffer = np.zeros(s_dim)
        self.s1_rec_buffer = np.zeros(s_dim)

        self.action = [0, 0]

    def handler_term(self):
        print("python program terminated usking Kill -15")
        if self.use_normalizer:
            print("save stats by kill -15")
            self.normalizer.save_stats()
        sys.exit(0)

    def network_monitor_per_ack(self, obs):
        self.obs = obs
        self.Uuid = obs[0] - 1
        self.ack_count += 1
        if not (self.obs[11]):
            self.loss_count += 1
        self.rtt_sum_ms += self.obs[9] / 1000
        self.min_rtt_monitor()
        self.srtt_monitor()
        print(self.Uuid, obs[2] * constants.US_TO_SEC, "rtt", self.obs[9] / 1000)

        return self.srtt_ms / 1000, self.min_rtt_ms * 1000

    def reset_network_status(self):
        self.ack_count = 0
        self.loss_count = 0
        self.rtt_sum_ms = 0

    def min_rtt_monitor(self):
        if not self.min_rtt_ms:
            self.min_rtt_ms = self.obs[9] / 1000
        else:
            self.min_rtt_ms = min(self.min_rtt_ms, self.obs[9] / 1000)

    def srtt_monitor(self):
        if self.srtt_ms:
            self.srtt_ms = (1 - constants.G) * self.srtt_ms + constants.G * self.obs[9] / 1000
        else:
            self.srtt_ms = self.obs[9] / 1000

    def is_this_new_mtp(self):
        self.new_mtp = int(self.obs[2] / constants.MTP) != self.timestamp

        return self.new_mtp

    def calculate_new_timestamp(self):
        self.timestamp = int(self.obs[2] / constants.MTP)

    def calculate_network_stats_per_mtp(self):
        self.interval_cnt = int(self.obs[2] / constants.MTP) - self.timestamp
        total_received_bits = self.ack_count * constants.MTU_BYTES * constants.BYTE_TO_BITS
        total_lost_bits = self.loss_count * constants.MTU_BYTES * constants.BYTE_TO_BITS
        total_time_passed = self.interval_cnt * constants.MTP * constants.US_TO_SEC
        self.throughput = total_received_bits / total_time_passed * constants.BPS_TO_MBPS
        self.loss_rate = total_lost_bits / total_time_passed * constants.BPS_TO_MBPS
        self.avg_rtt_ms = self.rtt_sum_ms / self.ack_count

    def print_network_status(self):
        for i in range(self.interval_cnt):
            print(self.Uuid, (self.timestamp + 1 + i) * 0.02, "throughput", self.throughput)
            print(self.Uuid, (self.timestamp + 1 + i) * 0.02, "loss_rate", self.loss_rate)
            print(self.Uuid, (self.timestamp + 1 + i) * 0.02, "srtt", self.srtt_ms)

    def reset(self, s_dim):

        self.max_bw = 0.0
        self.max_cwnd = 0.0
        self.max_smp = 0.0
        self.min_del = 9999999.0

        self.obs = np.zeros(15)
        self.z0 = np.zeros(15)

        # updating part
        self.alpha = 1
        self.last_agent_decision = 0
        self.ack_count = 0
        self.loss_count = 0
        self.rtt_sum_ms = 0
        self.min_rtt_ms = 0
        self.srtt_ms = 0
        self.timestamp = 0
        self.interval_cnt = 0
        self.throughput = 0
        self.loss_rate = 0
        self.avg_rtt_ms = 0
        self.epoch_for_whole = 0
        self.epoch_for_this_epi = 0
        self.reward = 0
        self.ret = 0
        self.Uuid = 0
        self.terminal = False
        self.error_code = True

        self.new_mtp = True

        self.s0 = np.zeros(s_dim)
        self.s1 = np.zeros(s_dim)
        self.s0_rec_buffer = np.zeros(s_dim)
        self.s1_rec_buffer = np.zeros(s_dim)

        self.action = [0, 0]

        state, delay_, rew0, error_code = self.get_state()

        return state

    def get_dims_info(self):
        return self.params.dict['state_dim'], self.params.dict['action_dim']

    def obs_to_state(self):

        self.z0[0] = self.avg_rtt_ms  # rtt_ms (orca-server-mahimahi.cc)
        self.z0[1] = self.throughput  # avg thr bps
        self.z0[2] = self.ack_count  # num of rtt samples used for averaging
        self.z0[3] = self.obs[2]/1000000  # time after simulation started
        # self.z0[4] = 0
        self.z0[5] = self.obs[5]  # cwnd_u
        self.z0[6] = self.obs[5] * 8 / (self.srtt_ms / 1000)  # pacing_rate
        self.z0[7] = self.loss_rate  # loss_rate
        self.z0[8] = self.srtt_ms  # srtt_ms
        # self.z0[9] = self.obs[4]  # snd_ssthresh
        # self.z0[10] = self.obs[8]  # packets_out
        # self.z0[11] = 0  # retrans_out
        # self.z0[12] = 0  # max packets_out in last window
        self.z0[13] = self.obs[6]  # mss
        self.z0[14] = self.min_rtt_ms  # min_rtt

    def get_state(self, evaluation=True):

        s0 = self.z0
        reward = 0
        state = np.zeros(1)

        if len(s0) == (self.params.dict['input_dim']):
            d = s0[0]
            # thr = s0[1]
            samples = s0[2]
            # delta_t = s0[3]
            # target_ = s0[4]
            cwnd = s0[5]
            # pacing_rate = s0[6]
            # loss_rate = s0[7]
            # srtt_ms = s0[8]
            # snd_ssthresh = s0[9]
            # packets_out = s0[10]
            # retrans_out = s0[11]
            # max_packets_out = s0[12]
            # mss = s0[13]
            min_rtt = s0[14]

            if self.use_normalizer:
                if not evaluation:
                    self.normalizer.observe(s0)
                s0 = self.normalizer.normalize(s0)
                min_ = self.normalizer.stats()
            else:
                min_ = s0-s0

            d_n = s0[0]-min_[0]
            # thr_n = s0[1]
            thr_n_min = s0[1]-min_[1]
            # samples_n = s0[2]
            samples_n_min = s0[2]-min_[2]
            delta_t_n = s0[3]
            # delta_t_n_min = s0[3]-min_[3]

            cwnd_n_min = s0[5]-min_[5]
            pacing_rate_n_min = s0[6]-min_[6]
            loss_rate_n_min = s0[7]-min_[7]
            srtt_ms_min = s0[8]-min_[8]
            # snd_ssthresh_min = s0[9]-min_[9]
            # packets_out_min = s0[10]-min_[10]
            # retrans_out_min = s0[11]-min_[11]
            # max_packets_out_min = s0[12]-min_[12]
            # mss_min = mss-min_[13]
            min_rtt_min = min_rtt-min_[14]

            if not self.use_normalizer:
                # thr_n = thr_n
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

            if min_rtt_min*(self.params.dict['delay_margin_coef']) < srtt_ms_min:  # out of delay budget
                delay_metric = (min_rtt_min*(self.params.dict['delay_margin_coef']))/srtt_ms_min
            else:  # within delay budget
                delay_metric = 1

            reward = (thr_n_min-5*loss_rate_n_min)/self.max_bw*delay_metric
            if self.max_bw != 0:
                state[0] = thr_n_min / self.max_bw
                tmp = pacing_rate_n_min / self.max_bw
                if tmp > 10:
                    tmp = 10
                state = np.append(state, [tmp])
                state = np.append(state, [5*loss_rate_n_min/self.max_bw])

            else:
                state[0] = 0
                state = np.append(state, [0])
                state = np.append(state, [0])
            if cwnd:
                state = np.append(state, [samples/cwnd])
            else:
                state = np.append(state, 0)
            state = np.append(state, [delta_t_n])
            if srtt_ms_min:
                state = np.append(state, [min_rtt_min / srtt_ms_min])
            else:
                state = np.append(state, 0)
            state = np.append(state, [delay_metric])
            # print(self.Uuid, self.obs[2] / 1000000, "STATE", state)
            return state, d, reward, True
        else:
            return state, 0.0, reward, False

    def states_and_reward_from_init(self, eval_=True):
        self.obs_to_state()
        self.s0, delay_, self.reward, error_code = self.get_state(evaluation=eval_)

    def states_and_reward_from_last_action(self, eval_=True):
        self.obs_to_state()
        self.s1, delay_, self.reward, error_code = self.get_state(evaluation=eval_)

    def concate_new_state(self):
        self.s1_rec_buffer = np.concatenate((self.s0_rec_buffer[self.params.dict['state_dim']:], self.s1))

        return self.s1_rec_buffer

    def init_state_history(self):
        self.s0_rec_buffer[-1 * self.params.dict['state_dim']:] = self.s0

    def epoch_count(self):
        self.epoch_for_whole += 1
        self.epoch_for_this_epi += 1

    def calculate_return(self):
        if self.epoch_for_this_epi <= 50:
            self.ret = (self.ret * (self.epoch_for_this_epi - 1) + self.reward) / self.epoch_for_this_epi
        else:
            self.ret = (1 - constants.RET_SMOOTHING_FACTOR) * self.ret + constants.RET_SMOOTHING_FACTOR * self.reward

    def print_orca_status(self):
        # print(self.Uuid, self.obs[2] * constants.US_TO_SEC, "reward", self.reward, "epochs", self.epoch_for_whole)
        # print(self.Uuid, self.obs[2] * constants.US_TO_SEC, "return", self.ret, "epochs", self.epoch_for_whole)
        # print(self.Uuid, self.obs[2] * constants.US_TO_SEC, "alpha", self.alpha)
        pass

    def update_states_history(self):
        self.s0 = self.s1
        self.s0_rec_buffer = self.s1_rec_buffer

    def agent_action_to_alpha(self, a1):
        self.last_agent_decision = a1
        self.alpha = math.pow(4, a1)

    def orca_over_cubic(self, cubic_action):
        self.action[1] = int(cubic_action[1] * self.alpha)
        self.action[0] = int(1.2 * self.action[1] * constants.BYTE_TO_BITS / (self.srtt_ms / 1000) * (self.obs[6] + constants.HEADER_SIZE) / self.obs[6])

        return self.action

class Normalizer:
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
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x-last_mean)*(x-self.mean)
        self.var = self.mean_diff/self.n

    def normalize(self, inputs):
        obs_std = np.sqrt(self.var)
        a = np.zeros(self.dim)
        if self.n > 2:
            a = (inputs - self.mean)/obs_std
            for i in range(0, self.dim):
                if a[i] < self.min[i]:
                    self.min[i] = a[i]
            return a
        else:
            return np.zeros(self.dim)

    def normalize_delay(self, delay):
        obs_std = math.sqrt(self.var[0])
        if self.n > 2:
            return (delay - self.mean[0])/obs_std
        else:
            return 0

    def stats(self):
        return self.min

    def save_stats(self):
        dic = {}
        dic['n'] = self.n
        dic['mean'] = self.mean.tolist()
        dic['mean_diff'] = self.mean_diff.tolist()
        dic['var'] = self.var.tolist()
        dic['min'] = self.min.tolist()
        import json
        with open(os.path.join(self.params.dict['train_dir'], 'stats.json'), 'w') as fp:
            json.dump(dic, fp)

        print("--------save stats at{}--------".format(self.params.dict['train_dir']))
        logger.info("--------save stats at{}--------".format(self.params.dict['train_dir']))

    def load_stats(self, file='stats.json'):
        import json
        if os.path.isfile(os.path.join(self.params.dict['train_dir'], file)):
            print("Stats exist!, load", "self.config.task")
            with open(os.path.join(self.params.dict['train_dir'], file), 'r') as fp:
                history_stats = json.load(fp)
                print(history_stats)
            self.n = history_stats['n']
            self.mean = np.asarray(history_stats['mean'])
            self.mean_diff = np.asarray(history_stats['mean_diff'])
            self.var = np.asarray(history_stats['var'])
            self.min = np.asarray(history_stats['min'])
            return True
        else:
            print("stats file is missing when loading")
            return False
