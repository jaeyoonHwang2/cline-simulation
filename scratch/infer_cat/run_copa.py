import logging
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from ns3gym import ns3env
from copa import copa_agent

print("pid", os.getpid())
tf.get_logger().setLevel(logging.ERROR)

# parameters config
port = 6465
num_senders = 3
done = False
info = None
mtu = 150
delta_ = [(1 / 2) for i in range(num_senders)]
initial_cwnd_ = [1 for i in range(num_senders)]

env = ns3env.Ns3Env(port=port, stepTime=0.000000001, startSim=0.00001, simSeed=12, debug=False)
copa_sender = [copa_agent(delta = delta_[i], initial_cwnd = initial_cwnd_[i]) for i in range(num_senders)]
obs = env.reset()

while True:

    Uuid = obs[0] - 1

    copa_sender[Uuid].collect_network_samples(obs)
    copa_sender[Uuid].monitor_network(obs, Uuid)
    copa_sender[Uuid].velocity_calculate(obs, Uuid)
    action = copa_sender[Uuid].get_action(obs)
    obs, rewardDontUse, done, info = env.step(action)

    if done: 
        
        done = False
        info = None
        break            

