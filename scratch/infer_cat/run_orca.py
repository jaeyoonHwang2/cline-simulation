import argparse
import tensorflow as tf
import os
import numpy as np
import random
import logging
import socket
import json
import time

from agent import Agent
from cubic import CubicAgent
from utils import Params
from envwrapper import TcpEnvWrapper
from ns3gym import ns3env
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.get_logger().setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--load', action='store_true', default=False, help='default is  %(default)s')
parser.add_argument('--eval', action='store_true', default=True, help='default is  %(default)s')
parser.add_argument('--mode', type=str, choices=['inference', 'training'], required=True, help='inference / training')
parser.add_argument('--role', type=str, choices=['learner', 'actor'], required=True, help='learner / actor')
parser.add_argument('--task', type=int, required=True)

# parameters for config
sim_time = 20
start_sim = 0.0001
step_time = 0.0001

global config
config = parser.parse_args()

if config.mode == 'inference':
    base_path = '../../../../../infer_learner'
else:  # 'training
    base_path = '../../../../../train_learner'
    HOST = '127.0.0.1'
    PORT = 9999
    actor_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    actor_socket.connect((HOST, PORT))
    print("Actor socket connected to learner")

port = 5000 + config.task

# parameters from parser
global params
params = Params(os.path.join(base_path, 'params.json'))

if config.mode == 'inference':
    simArgs = {"--duration": sim_time, }
else:
    simArgs = {"--duration": params.dict['simTime'], }

# monitoring variables
num_senders = 1

if config.mode == 'inference':
    local_job_device = ''
    shared_job_device = ''
    def is_actor_fn(i): return True
    global_variable_device = '/gpu'
    is_learner = False
    server = tf.train.Server.create_local_server()
    filters = []

else:  # 'training'
    local_job_device = '/job:%s/task:%d' % (config.role, config.task)
    shared_job_device = '/job:learner/task:0'
    is_learner = config.role == 'learner'
    global_variable_device = shared_job_device + '/cpu'
    def is_actor_fn(i): return config.role == 'actor' and i == config.task
    cluster = tf.train.ClusterSpec({
            'actor': ['localhost:%d' % (8001 + i) for i in range(params.dict['num_actors'])],
            'learner': ['localhost:8000']
        })
    server = tf.train.Server(cluster, job_name=config.role, task_index=config.task)
    filters = [shared_job_device, local_job_device]

s_dim = 0
env_peek = TcpEnvWrapper(s_dim, params, use_normalizer=params.dict['use_normalizer'])
s_dim, a_dim = env_peek.get_dims_info()
params.dict['state_dim'] = s_dim
s_dim = s_dim * params.dict['rec_dim']

with tf.Graph().as_default(), tf.device(local_job_device + '/cpu'):

    tf.set_random_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    actor_op = []
    tfeventdir = os.path.join(base_path, params.dict['logdir'], config.role+str(0))
    params.dict['train_dir'] = tfeventdir

    if not os.path.exists(tfeventdir):
        os.makedirs(tfeventdir)
    summary_writer = tf.summary.FileWriterCache.get(tfeventdir)

    with tf.device(shared_job_device):
        agent = Agent(s_dim, a_dim, batch_size=params.dict['batch_size'], summary=summary_writer, h1_shape=params.dict['h1_shape'],
                    h2_shape=params.dict['h2_shape'], stddev=params.dict['stddev'], mem_size=params.dict['memsize'], gamma=params.dict['gamma'],
                    lr_c=params.dict['lr_c'], lr_a=params.dict['lr_a'], tau=params.dict['tau'], PER=params.dict['PER'], CDQ=params.dict['CDQ'],
                    LOSS_TYPE=params.dict['LOSS_TYPE'], noise_type=params.dict['noise_type'], noise_exp=params.dict['noise_exp'])
        cubic_senders = [CubicAgent() for i in range(num_senders)]
        dtypes = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
        shapes = [[s_dim], [a_dim], [1], [s_dim], [1]]
        queue = tf.FIFOQueue(10000, dtypes, shapes, shared_name="rp_buf")

        env = [TcpEnvWrapper(s_dim, params, use_normalizer=params.dict['use_normalizer']) for i in range(num_senders)]
        envNs3 = ns3env.Ns3Env(port=port, stepTime=step_time, startSim=start_sim, simSeed=12, simArgs=simArgs, debug=False)
    
    for i in range(params.dict['num_actors']):
    
        if is_actor_fn(i):
    
            a_s0 = tf.placeholder(tf.float32, shape=[s_dim], name='a_s0')
            a_action = tf.placeholder(tf.float32, shape=[a_dim], name='a_action')
            a_reward = tf.placeholder(tf.float32, shape=[1], name='a_reward')
            a_s1 = tf.placeholder(tf.float32, shape=[s_dim], name='a_s1')
            a_terminal = tf.placeholder(tf.float32, shape=[1], name='a_terminal')
            a_buf = [a_s0, a_action, a_reward, a_s1, a_terminal]

            with tf.device(shared_job_device):
                actor_op.append(queue.enqueue(a_buf))

    params.dict['ckptdir'] = os.path.join(base_path, params.dict['ckptdir'])
    print("## checkpoint dir:", params.dict['ckptdir'])
    isckpt = os.path.isfile(os.path.join(params.dict['ckptdir'], 'checkpoint'))
    print("## checkpoint exists?:", isckpt)
    if not isckpt:
        print("\n# # # # # # Warning ! ! ! No checkpoint is loaded, use random model! ! ! # # # # # #\n")

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    
    if config.mode == 'inference':
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
    
    iteration = 0

    while True:  # training -> lots of episodes

        obs = envNs3.reset()
        done = False
        info = None

        slow_start = [True for i in range(num_senders)]

        for i in range(num_senders):
            env[i].reset(s_dim)
            env[i].init_state_history()
            a_init = agent.get_action(env[i].s0_rec_buffer, not config.eval)
            env[i].agent_action_to_alpha(a_init)

        while True:  # inference -> single episode

            Uuid = obs[0] - 1

            if not obs[11]:
                slow_start[Uuid] = False

            srtt_sec, min_rtt_us = env[Uuid].network_monitor_per_ack(obs, iteration, sim_time)

            if env[Uuid].is_this_new_mtp():  # new MTP
                env[Uuid].calculate_network_stats_per_mtp()
                env[Uuid].print_network_status()
                env[Uuid].calculate_new_timestamp()

                if not slow_start[Uuid]:  # no slow start
                    env[Uuid].epoch_count()
                    env[Uuid].states_and_reward_from_last_action()
                    env[Uuid].calculate_return()
                    env[Uuid].concate_new_state()
                    a1 = agent.get_action(env[Uuid].s1_rec_buffer, not config.eval)
                    fd = {a_s0: env[Uuid].s0_rec_buffer, a_action: env[Uuid].last_agent_decision, a_reward: np.array([env[Uuid].reward]),
                        a_s1: env[Uuid].s1_rec_buffer, a_terminal: np.array([env[Uuid].terminal], np.float)}
                    if not config.eval:
                        mon_sess.run(actor_op, feed_dict=fd)

                    if config.mode == 'training':
                        if ((env[Uuid].epoch_for_whole % params.dict['numSample']) == 0):
                            actor_socket.send(json.dumps({"Wait": 1}).encode())
                            print("Waiting for learner to update", time.time())
                            json.loads(actor_socket.recv(1024).decode()).get("Free")
                            print("Learner has updated", time.time())

                    env[Uuid].agent_action_to_alpha(a1)
                    env[Uuid].update_states_history()
                    env[Uuid].print_orca_status()
                    env[Uuid].reset_network_status()
                else:
                    env[Uuid].reset_network_status()
            cubic_action = cubic_senders[Uuid].get_action(obs, srtt_sec, min_rtt_us)

            if env[Uuid].epoch_for_this_epi:  # no slow start
                orca_action = env[Uuid].orca_over_cubic(cubic_action)
            else:
                orca_action = cubic_action

            print(Uuid, obs[2] / 1000000 + iteration * sim_time, "CUBIC_CWND",cubic_action[1])
            print(Uuid, obs[2] / 1000000 + iteration * sim_time, "ORCA_ALPHA", env[Uuid].alpha)
            print(Uuid, obs[2] / 1000000 + iteration * sim_time, "ORCA_CWND", orca_action[1])

            obs, reward, done, info = envNs3.step(orca_action)

            if done:
                done = False
                info = None
                print("An episode is over")
                iteration += 1
                break
        
        if config.mode == 'inference':
                break
