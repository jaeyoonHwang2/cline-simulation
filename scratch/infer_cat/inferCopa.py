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
num_senders = 1
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

