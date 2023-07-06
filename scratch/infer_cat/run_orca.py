from agent import Agent
import os

from cubic import CubicAgent
import numpy as np
import random
import pickle
from utils import Params
from ns3gym import ns3env
import math
import argparse
from orca import OrcaAgent

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, choices=['inference', 'training'], required=True) # inference or training
parser.add_argument('--role', type=str, choices=['learner', 'actor'], required=True) # actor or learner
parser.add_argument('--task', type=int, required=True)
global config
config = parser.parse_args()

port = 5555+config.task
stepTime = 0.000001
startSim = 0.00001
simArgs = {"--duration": 50}

num_senders = 1



orca_worker = OrcaAgent()
orca_worker.init_config(config)

with tf.Graph().as_default(),\
    tf.device(local_job_device + '/cpu'):





    while True:
















    tf.set_random_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    actor_op = []
    check_op = []
    tfeventdir = os.path.join( base_path, params.dict['logdir'], job_name+str(task) )
    params.dict['train_dir'] = tfeventdir

    if not os.path.exists(tfeventdir):
        os.makedirs(tfeventdir)
    summary_writer = tf.summary.FileWriterCache.get(tfeventdir)

    with tf.device(shared_job_device):

        agent = Agent(s_dim, a_dim, batch_size=params.dict['batch_size'], summary=summary_writer,h1_shape=params.dict['h1_shape'],
                    h2_shape=params.dict['h2_shape'],stddev=params.dict['stddev'],mem_size=params.dict['memsize'],gamma=params.dict['gamma'],
                    lr_c=params.dict['lr_c'],lr_a=params.dict['lr_a'],tau=params.dict['tau'],PER=params.dict['PER'],CDQ=params.dict['CDQ'],
                    LOSS_TYPE=params.dict['LOSS_TYPE'],noise_type=params.dict['noise_type'],noise_exp=params.dict['noise_exp'])
        cubicAgent = [CubicAgent() for i in range(numSenders)]

        dtypes = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
        shapes = [[s_dim], [a_dim], [1], [s_dim], [1]]
        queue = tf.FIFOQueue(10000, dtypes, shapes, shared_name="rp_buf")

    if is_learner:
        with tf.device(params.dict['device']):
            agent.build_learn()

            agent.create_tf_summary()

        if params.dict['load']==True and params.dict['eval']==False:
            if os.path.isfile(os.path.join(params.dict['train_dir'], "replay_memory.pkl")):
                with open(os.path.join(params.dict['train_dir'], "replay_memory.pkl"), 'rb') as fp:
                    replay_memory = pickle.load(fp)

    for i in range(params.dict['num_actors']):
        if is_actor_fn(i): # if actor
            env = TCP_Env_Wrapper("TCP", params, config=config, for_init_only=False, use_normalizer=params.dict['use_normalizer'])
            envNs3 = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=12, simArgs=simArgs, debug=True)

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
    a[Uuid] = agent.get_action(s0_rec_buffer[Uuid], not params.dict['eval'])
    a[Uuid] = a[Uuid][0][0]

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
                        if params.dict['recurrent']: a_ = agent.get_action(s0_rec_buffer[Uuid], not params.dict['eval'])
                        else: a_ = agent.get_action(s0[Uuid], not params.dict['eval'])
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

                        if error_code[Uuid] == True:
                        
                            s1_rec_buffer[Uuid] = np.concatenate( (s0_rec_buffer[Uuid][params.dict['state_dim']:], s1[Uuid]) )
                            a1 = agent.get_action(s1_rec_buffer[Uuid], not params.dict['eval'])

                            # action is a1 (later used to calculate coefficient k = 4^a1)
                            a1 = a1[0][0]
                            
                        else: 

                            print("TaskID:"+str(task)+"Invalid state received...\n")
                            continue
                                    
                        if params.dict['recurrent']: fd = {a_s0:s0_rec_buffer[Uuid], a_action:a[Uuid], a_reward:np.array([r[Uuid]]), a_s1:s1_rec_buffer[Uuid], a_terminal:np.array([terminal[Uuid]], np.float)}
                        else: fd = {a_s0:s0[Uuid], a_action:a[Uuid], a_reward:np.array([r[Uuid]]), a_s1:s1[Uuid], a_terminal:np.array([terminal[Uuid]], np.float)}

                        if not params.dict['eval']: mon_sess.run(actor_op, feed_dict=fd)

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
                    if (action[Uuid][1] > 100000): action[Uuid][1] = 100000
                    if (action[Uuid][1] < 180): action[Uuid][1] = 180
                    action[Uuid][0] = int(1.2 * max(action[Uuid][1], obs[8]) * 8 / (srtt[Uuid] / 1000) * (obs[6] + 60) / obs[6]) # pacingRate: mss -> mtu
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
