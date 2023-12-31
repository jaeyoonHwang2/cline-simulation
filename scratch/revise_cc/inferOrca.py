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

from telnetlib import DM
import logging
import tensorflow as tf
from agent import Agent
import os

from tcpCubic import TcpCubic
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import numpy as np
import random
import datetime
from utils import logger, Params
from envwrapper import Env_Wrapper, TCP_Env_Wrapper
from ns3gym import ns3env
import math

tf.get_logger().setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--load', action='store_true', default=False, help='default is  %(default)s')
parser.add_argument('--eval', action='store_true', default=True, help='default is  %(default)s')
base_path = '../../../../../infer_learner'
job_name = 'actor'  


# parameters from parser
global config
global params
config = parser.parse_args()

# parameters from .json file
params = Params(os.path.join(base_path,'params.json'))
seed = 12 
debug = False  

# parameters for config
startSim = 0.00001
port = 7996
stepTime = 0.000001
simTime = 20
simArgs = {"--duration": simTime, }

# monitoring variables
numSenders = 1
mtp = 20000
iteration = 0

alpha = [1 for i in range (numSenders)]
ack_count = [0 for i in range (numSenders)] 
loss_count = [0 for i in range (numSenders)]
rtt_sum = [0 for i in range (numSenders)] 
min_rtt = [0 for i in range (numSenders)]
srtt = [0 for i in range (numSenders)]
timestamp = [0 for i in range (numSenders)] 
interval_cnt = [0 for i in range (numSenders)]
throughput = [0 for i in range (numSenders)]
loss_rate = [0 for i in range (numSenders)]
rtt = [0 for i in range (numSenders)]
epoch = [0 for i in range (numSenders)]
epoch_ = [0 for i in range (numSenders)]
r = [0 for i in range (numSenders)]
ret = [0 for i in range (numSenders)]

local_job_device = ''
shared_job_device = ''
def is_actor_fn(i): return True
global_variable_device = '/gpu'
is_learner = False
server = tf.train.Server.create_local_server()
filters = []

env_peek = TCP_Env_Wrapper("TCP", params,use_normalizer=params.dict['use_normalizer'])

s_dim, a_dim = env_peek.get_dims_info()
action_scale, action_range = env_peek.get_action_info()

params.dict['state_dim'] = s_dim
s_dim = s_dim * params.dict['rec_dim']

with tf.Graph().as_default(),\
    tf.device(local_job_device + '/cpu'):

    tf.set_random_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    actor_op = []
    check_op = []
    now = datetime.datetime.now()
    tfeventdir = os.path.join( base_path, params.dict['logdir'], job_name+str(0) )
    params.dict['train_dir'] = tfeventdir

    if not os.path.exists(tfeventdir):
        os.makedirs(tfeventdir)
    summary_writer = tf.summary.FileWriterCache.get(tfeventdir)

    with tf.device(shared_job_device):

        agent = Agent(s_dim, a_dim, batch_size=params.dict['batch_size'], summary=summary_writer,h1_shape=params.dict['h1_shape'],
                    h2_shape=params.dict['h2_shape'],stddev=params.dict['stddev'],mem_size=params.dict['memsize'],gamma=params.dict['gamma'],
                    lr_c=params.dict['lr_c'],lr_a=params.dict['lr_a'],tau=params.dict['tau'],PER=params.dict['PER'],CDQ=params.dict['CDQ'],
                    LOSS_TYPE=params.dict['LOSS_TYPE'],noise_type=params.dict['noise_type'],noise_exp=params.dict['noise_exp'])
        cubicAgent = [TcpCubic() for i in range(numSenders)]

        dtypes = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
        shapes = [[s_dim], [a_dim], [1], [s_dim], [1]]
        queue = tf.FIFOQueue(10000, dtypes, shapes, shared_name="rp_buf")

        env = TCP_Env_Wrapper(0, params, config=config, for_init_only=False, use_normalizer=params.dict['use_normalizer']) 
        envNs3 = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)

        a_s0 = tf.placeholder(tf.float32, shape=[s_dim], name='a_s0') 
        a_action = tf.placeholder(tf.float32, shape=[a_dim], name='a_action')
        a_reward = tf.placeholder(tf.float32, shape=[1], name='a_reward') 
        a_s1 = tf.placeholder(tf.float32, shape=[s_dim], name='a_s1') 
        a_terminal = tf.placeholder(tf.float32, shape=[1], name='a_terminal') 
        a_buf = [a_s0, a_action, a_reward, a_s1, a_terminal]

        with tf.device(shared_job_device):
            actor_op.append(queue.enqueue(a_buf))

    params.dict['ckptdir'] = os.path.join( base_path, params.dict['ckptdir'])
    print("## checkpoint dir:", params.dict['ckptdir'])
    isckpt = os.path.isfile(os.path.join(params.dict['ckptdir'], 'checkpoint') )
    print("## checkpoint exists?:", isckpt)
    if isckpt== False:
        print("\n# # # # # # Warning ! ! ! No checkpoint is loaded, use random model! ! ! # # # # # #\n")

    tfconfig = tf.ConfigProto(allow_soft_placement=True)

    mon_sess = tf.train.SingularMonitoredSession(checkpoint_dir=params.dict['ckptdir'])

    agent.assign_sess(mon_sess)

    obs = envNs3.reset()
    
    Uuid = obs[0] - 1

    done = False
    info = None

    # init. variables

    alpha = [1 for i in range (numSenders)]
    a = [0 for i in range (numSenders)]
    ack_count = [0 for i in range (numSenders)] 
    loss_count = [0 for i in range (numSenders)]
    rtt_sum = [0 for i in range (numSenders)] 
    min_rtt = [0 for i in range (numSenders)]
    srtt = [0 for i in range (numSenders)]
    timestamp = [0 for i in range (numSenders)] 
    interval_cnt = [0 for i in range (numSenders)]
    throughput = [0 for i in range (numSenders)]
    loss_rate = [0 for i in range (numSenders)]
    rtt = [0 for i in range (numSenders)]
    epoch = [0 for i in range (numSenders)]
    epoch_ = [0 for i in range (numSenders)]
    r = [0 for i in range (numSenders)]
    ret = [0 for i in range (numSenders)]
    action = [[0, 0] for i in range (numSenders)]
    orca_init = [True for i in range (numSenders)]
    slow_start = [True for i in range (numSenders)]
    terminal = [False for i in range (numSenders)]
    error_code = [True for i in range (numSenders)]

    s0 = [np.zeros([s_dim]) for i in range (numSenders)]
    s1 = [np.zeros([s_dim]) for i in range (numSenders)]
    s0_rec_buffer = [np.zeros([s_dim]) for i in range (numSenders)]
    s1_rec_buffer = [np.zeros([s_dim]) for i in range (numSenders)]
    a_ = agent.get_action(s0_rec_buffer[Uuid], not config.eval)    
    a[Uuid] = a_[0][0]
    
    while True: # one iteration
    
        Uuid = obs[0] - 1
        
        # increase counts
        ack_count[Uuid] += 1
        if not (obs[11]): loss_count[Uuid] += 1
        rtt_sum[Uuid] += obs[9] / 1000

        if not (min_rtt[Uuid]): min_rtt[Uuid] = obs[9] / 1000 # min_rtt_ms
        else: min_rtt[Uuid] = min(min_rtt[Uuid], obs[9] / 1000)

        if (srtt[Uuid]): srtt[Uuid] = 7/8 * srtt[Uuid] + 1/8 * obs[9] / 1000 # srtt_ms 
        else: srtt[Uuid] = obs[9] / 1000

        if (obs[7]): # ack is recognized            

            # mtp has passed from the last report 
            # -> Orca will take into action
            if (int(obs[2] / mtp) != timestamp[Uuid]):

                if not (timestamp[Uuid]): timestamp[Uuid] = int(obs[2] / mtp) - 1
                
                interval_cnt[Uuid] = int(obs[2] / mtp) - timestamp[Uuid]

                # throughput / loss rate / rtt is calculated
                throughput[Uuid] = ack_count[Uuid] * 1500 * 8 / (interval_cnt[Uuid] * (mtp / (1000 * 1000))) / (1024 * 1024) # Mbps
                loss_rate[Uuid] = loss_count[Uuid] * 1500 * 8 / (interval_cnt[Uuid] * (mtp / (1000 * 1000))) / (1024 * 1024) # Mbps
                rtt[Uuid] = rtt_sum[Uuid] / ack_count[Uuid] # ms

                # print throughput / loss rate / rtt
                for i in range(interval_cnt[Uuid]):
                    print (Uuid, (timestamp[Uuid] + 1 + i) * 0.02, "throughput", throughput[Uuid])
                    print (Uuid, (timestamp[Uuid] + 1 + i) * 0.02, "loss_rate", loss_rate[Uuid])
                    print (Uuid, (timestamp[Uuid] + 1 + i) * 0.02, "srtt", srtt[Uuid])
                
                if not (slow_start[Uuid]):                                            

                    if (orca_init[Uuid]):
                        orca_init[Uuid] = False

                        s0[Uuid] = env.reset(obs, throughput[Uuid], rtt[Uuid], loss_rate[Uuid], srtt[Uuid], min_rtt[Uuid], ack_count[Uuid], iteration, simTime)
                        s0_rec_buffer[Uuid] = np.zeros([s_dim]) 
                        s1_rec_buffer[Uuid] = np.zeros([s_dim]) 
                        s0_rec_buffer[Uuid][-1*params.dict['state_dim']:] = s0[Uuid]
                        if params.dict['recurrent']: a_ = agent.get_action(s0_rec_buffer[Uuid], not config.eval)                        
                        else: a_ = agent.get_action(s0[Uuid], not config.eval)
                        a[Uuid] = a_[0][0]
                        alpha[Uuid] = math.pow(4, a[Uuid])

                    else:

                        # Orca action
                        # epoch: total epochs / epoch_: epochs for this episode
                        epoch[Uuid] += 1
                        epoch_[Uuid] += 1
                        
                        # return state and reward & erase used states
                        s1[Uuid], r[Uuid], terminal[Uuid], error_code[Uuid] = env.step(obs, throughput[Uuid], rtt[Uuid], loss_rate[Uuid], srtt[Uuid], min_rtt[Uuid], ack_count[Uuid], iteration, simTime, eval_= eval)

                        if (epoch_[Uuid] <= 50): ret[Uuid] = (ret[Uuid] * (epoch_[Uuid] - 1) + r[Uuid]) / epoch_[Uuid]
                        else: ret[Uuid] = 0.98 * ret[Uuid] + 0.02 * r[Uuid]

                        # print reward / return                    
                        print (Uuid, int(obs[2] / mtp) * 0.02, "reward", r[Uuid], "epochs", epoch[Uuid])
                        print (Uuid, int(obs[2] / mtp) * 0.02 , "return", ret[Uuid], "epochs", epoch[Uuid])                    
                    
                        s1_rec_buffer[Uuid] = np.concatenate( (s0_rec_buffer[Uuid][params.dict['state_dim']:], s1[Uuid]) )
                        if params.dict['recurrent']: a1 = agent.get_action(s1_rec_buffer[Uuid], not config.eval)
                        else: a1 = agent.get_action(s1[Uuid], not config.eval)
                        
                        # action is a1 (later used to calculate coefficient k = 4^a1)
                        a1 = a1[0][0]                            
                                    
                        fd = {a_s0:s0_rec_buffer[Uuid], a_action:a[Uuid], a_reward:np.array([r[Uuid]]), a_s1:s1_rec_buffer[Uuid], a_terminal:np.array([terminal[Uuid]], np.float)}                        

                        if not config.eval: mon_sess.run(actor_op, feed_dict=fd)      

                        s0[Uuid] = s1[Uuid]
                        a[Uuid] = a1
                        alpha[Uuid] = math.pow(4, a1)
                        print (Uuid, int(obs[2] / mtp) * 0.02, "orca_alpha", alpha[Uuid])                  

                    if params.dict['recurrent']: s0_rec_buffer[Uuid] = s1_rec_buffer[Uuid]

                    if not params.dict['use_TCP'] and (terminal[Uuid]):
                        if agent.actor_noise != None:
                            agent.actor_noise.reset()

                    # just to check CUBIC logic
                    ##alpha[Uuid] = 1

                    action[Uuid][1] = int(obs[5] * alpha[Uuid])
                    #if (action[Uuid][1] > 100000): action[Uuid][1] = 100000
                    if not (srtt[Uuid]): srtt[Uuid] = 120
                    if (action[Uuid][1] < 180): action[Uuid][1] = 180                
                    action[Uuid][0] = int(1.2 * action[Uuid][1] * 8 / (srtt[Uuid] / 1000) * (obs[6] + 60) / obs[6]) # pacingRate: mss -> mtu
                    print (Uuid, (obs[2] / 1000000), "orca_cwnd", action[Uuid][1])
                    
                    ack_count[Uuid] = 0
                    loss_count[Uuid] = 0
                    rtt_sum[Uuid] = 0

                    timestamp[Uuid] = int(obs[2] / mtp)

                    obs, reward, done, info = envNs3.step(action[Uuid])

                
                else: # slow start
                
                    action[Uuid], slow_start[Uuid] = cubicAgent[Uuid].get_action(obs, srtt[Uuid], min_rtt[Uuid], done, info)
                    print (Uuid, (obs[2] / 1000000), "cubic_cwnd", action[Uuid][1])
                    
                    ack_count[Uuid] = 0
                    loss_count[Uuid] = 0
                    rtt_sum[Uuid] = 0

                    timestamp[Uuid] = int(obs[2] / mtp)

                    obs, reward, done, info = envNs3.step(action[Uuid])

            else: # normal CUBIC
            
                action[Uuid], slow_start[Uuid] = cubicAgent[Uuid].get_action(obs, srtt[Uuid], min_rtt[Uuid], done, info)
                print (Uuid, (obs[2] / 1000000), "cubic_cwnd", action[Uuid][1])

                obs, reward, done, info = envNs3.step(action[Uuid])                    

        else: # no ack (probably loss)
        
            action[Uuid], slow_start[Uuid] = cubicAgent[Uuid].get_action(obs, srtt[Uuid], min_rtt[Uuid], done, info)
            print (Uuid, (obs[2] / 1000000), "cubic_cwnd", action[Uuid][1])

            obs, reward, done, info = envNs3.step(action[Uuid])


        if done: 

            done = False
            info = None
            iteration += 1 
            print ("An episode is over")
            break
