import constants
import tensorflow as tf
import os
import random
import numpy as np
import pickle

from agent import Agent
from cubic import CubicAgent
from envwrapper import TCP_Env_Wrapper
from utils import Params



class OrcaAgent:

    def __init__(self):
        # initialize config parameters (default is evaluation)
        self.s_dim = 0
        self.a_dim = 0
        self.local_job_device = ''
        self.shared_job_device = ''
        self.global_variable_device = '/gpu'
        self.is_learner = False
        self.server = tf.train.Server.create_local_server()
        self.filters = []
        self.env = 0
        self.envNs3 = 0
        self.cubicAgents = []

        # initalized monitoring parameters
        self.iteration = 0
        self.alpha = 1
        self.ack_count = 0
        self.loss_count = 0
        self.rtt_sum = 0
        self.min_rtt = 0
        self.srtt = 0
        self.timestamp = 0
        self.interval_cnt = 0
        self.throughput = 0
        self.loss_rate = 0
        self.rtt = 0
        self.epoch = 0
        self.epoch_ = 0 # need renaming
        self.r = 0 # need renaming
        self.ret = 0 # need renaming

    def init_config(self, config):
        if config.mode == 'inference':
            self.infer_config(config)
        elif config.mode == 'training':
            self.train_config(config)

        env_peek = TCP_Env_Wrapper("TCP", self.params, use_normalizer=self.params.dict['use_normalizer'])
        self.s_dim, self.a_dim = env_peek.get_dims_info()
        self.s_dim = self.s_dim * self.params.dict['rec_dim']
        if self.params.dict['use_hard_target'] == True:
            self.params.dict['tau'] = 1.0

    def infer_config(self, config):
        self.base_path = '../../../../../infer_learner'
        self.params = Params(os.path.join(self.base_path, 'params.json'))

    def train_config(self, config):
        self.base_path = '../../../../../train_learner'
        self.params = Params(os.path.join(self.base_path, 'params.json'))
        self.local_job_device = '/job:%s/task:%d' % (config.role, config.task)
        self.shared_job_device = '/job:learner/task:0'
        self.is_learner = config.role == 'learner'
        self.global_variable_device = self.shared_job_device + '/gpu'
        self.cluster = tf.train.ClusterSpec({
            'actor': ['localhost:%d' % (8001 + i) for i in range(self.params.dict['num_actors'])],
            'learner': ['localhost:8000']
        })
        self.server = tf.train.Server(self.cluster, job_name=config.role, task_index=config.task)
        self.filters = [self.shared_job_device, self.local_job_device]

    def on_run(self, config, num_senders):
        with tf.Graph().as_default(), \
            tf.device(self.local_job_device + '/cpu'):

            tf.set_random_seed(1234)
            random.seed(1234)
            np.random.seed(1234)

            actor_op = []
            tfeventdir = os.path.join(config.base_path, self.params.dict['logdir'], config.job_name + str(config.task))
            self.params.dict['train_dir'] = tfeventdir

            if not os.path.exists(tfeventdir):
                os.makedirs(tfeventdir)
            summary_writer = tf.summary.FileWriterCache.get(tfeventdir)

            with tf.device(self.shared_job_device):
                agent = Agent(s_dim, a_dim, batch_size=self.params.dict['batch_size'], summary=summary_writer,
                      h1_shape=self.params.dict['h1_shape'],
                      h2_shape=self.params.dict['h2_shape'], stddev=self.params.dict['stddev'],
                      mem_size=self.params.dict['memsize'], gamma=self.params.dict['gamma'],
                      lr_c=self.params.dict['lr_c'], lr_a=self.params.dict['lr_a'], tau=self.params.dict['tau'],
                      PER=self.params.dict['PER'], CDQ=self.params.dict['CDQ'],
                      LOSS_TYPE=self.params.dict['LOSS_TYPE'], noise_type=self.params.dict['noise_type'],
                      noise_exp=self.params.dict['noise_exp'])
                self.cubicAgents = [CubicAgent() for i in range(num_senders)]

                dtypes = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
                shapes = [[self.s_dim], [self.a_dim], [1], [self.s_dim], [1]]
                queue = tf.FIFOQueue(10000, dtypes, shapes, shared_name="rp_buf")

            if self.is_learner:
                with tf.device(self.params.dict['device']):
                    agent.build_learn()

                    agent.create_tf_summary()

                if self.params.dict['load']==True and self.params.dict['eval']==False:
                    if os.path.isfile(os.path.join(self.params.dict['train_dir'], "replay_memory.pkl")):
                        with open(os.path.join(self.params.dict['train_dir'], "replay_memory.pkl"), 'rb') as fp:
                            replay_memory = pickle.load(fp)

            for i in range(self.params.dict['num_actors']):
                if config.role == 'actor': # if actor
                    self.env = TCP_Env_Wrapper("TCP", params, config=config, for_init_only=False, use_normalizer=params.dict['use_normalizer'])
                    self.envNs3 = self.ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=12, simArgs=simArgs, debug=True)

                    a_s0 = tf.placeholder(tf.float32, shape=[s_dim], name='a_s0')
                    a_action = tf.placeholder(tf.float32, shape=[a_dim], name='a_action')
                    a_reward = tf.placeholder(tf.float32, shape=[1], name='a_reward')
                    a_s1 = tf.placeholder(tf.float32, shape=[s_dim], name='a_s1')
                    a_terminal = tf.placeholder(tf.float32, shape=[1], name='a_terminal')
                    a_buf = [a_s0, a_action, a_reward, a_s1, a_terminal]

                    with tf.device(shared_job_device):
                        actor_op.append(queue.enqueue(a_buf))

            if is_learner:
                Dequeue_Length = params.dict['dequeue_length']
                dequeue = queue.dequeue_many(Dequeue_Length)

            queuesize_op = queue.size()

            if params.dict['ckptdir'] is not None:
                params.dict['ckptdir'] = os.path.join( base_path, params.dict['ckptdir'])
                print("## checkpoint dir:", params.dict['ckptdir'])
                isckpt = os.path.isfile(os.path.join(params.dict['ckptdir'], 'checkpoint') )
                print("## checkpoint exists?:", isckpt)
                if isckpt== False:
                    print("\n# # # # # # Warning ! ! ! No checkpoint is loaded, use random model! ! ! # # # # # #\n")
            else:
                params.dict['ckptdir'] = tfeventdir

            tfconfig = tf.ConfigProto(allow_soft_placement=True)

            if params.dict['single_actor_eval']:
                mon_sess = tf.train.SingularMonitoredSession(
                    checkpoint_dir=params.dict['ckptdir'])
            else:
                mon_sess = tf.train.MonitoredTrainingSession(master=server.target,
                        save_checkpoint_secs=15,
                        save_summaries_secs=None,
                        save_summaries_steps=None,
                        is_chief=is_learner,
                        checkpoint_dir=params.dict['ckptdir'],
                        config=tfconfig,
                        hooks=None)

            agent.assign_sess(mon_sess)















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
                print(self.Uuid, self.simTime_us * constants.US_TO_SEC, "cwnd", self.last_cwnd_decision)

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
