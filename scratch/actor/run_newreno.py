from newreno import TcpNewReno
from monitor import monitoring_agent

from ns3gym import ns3env


# config
port = 6062
stepTime = 0.000001
startSim = 0.00001
simArgs = {"--duration": 30,}
num_senders = 1

# variables
done = False
info = None

# prepare environments
envNs3 = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=12, simArgs=simArgs, debug=False)
newreno_senders = [TcpNewReno() for i in range(num_senders)]
monitoring_agents = [monitoring_agent("newreno") for i in range(num_senders)]
obs = envNs3.reset()


while True: 

    Uuid = obs[0] - 1

    monitoring_agents[Uuid].monitor_network(obs)
    actions = newreno_senders[Uuid].get_action(obs)

    obs, reward, done, info = envNs3.step(actions)

    if done:

        done = False
        info = None
        break
