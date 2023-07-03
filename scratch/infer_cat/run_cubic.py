from cubic import CubicAgent
from monitor import MonitoringAgent

from ns3gym import ns3env
    
# config
port = 5555
stepTime = 0.000001
startSim = 0.00001
simArgs = {"--duration": 100}
num_senders = 3

# variables
done = False
info = None

# envNs3 = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=12, simArgs=simArgs, debug=False)
envNs3 = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=12, simArgs=simArgs, debug=True)
cubic_senders = [CubicAgent() for i in range(num_senders)]
monitoring_agents = [MonitoringAgent("cubic") for i in range(num_senders)]
obs = envNs3.reset()


while True: 

    Uuid = obs[0] - 1

    srtt_sec, min_rtt_us = monitoring_agents[Uuid].monitor_network(obs)
    actions = cubic_senders[Uuid].get_action(obs, srtt_sec, min_rtt_us)

    if not obs[11]:  # for loss debug
        obs, reward, done, info = envNs3.step(actions)
    else:
        obs, reward, done, info = envNs3.step(actions)

    if done:

        done = False
        info = None
        break
