    # unique socket ID
    socketUuid = obs[0]
    # TCP env type: event-based = 0 / time-based = 1
    envType = obs[1]
    # sim time in us
    simTime_us = obs[2]
    # unique node ID
    nodeId = obs[3]
    # current ssThreshold
    ssThresh = obs[4]
    # current contention window size
    cWnd = obs[5]
    # segment size
    segmentSize = obs[6]
    # number of acked segments
    segments_acked = obs[7]
    # estimated bytes in flight
    bytesInFlight  = obs[8]
    # last estimation of RTT
    lastRtt_us  = obs[9]
    # min value of RTT
    minRtt_us  = obs[10]
    # function from Congestion Algorithm (CA) interface:
    #  GET_SS_THRESH = 0 (packet loss),
    #  INCREASE_WINDOW (packet acked),
    #  PKTS_ACKED (unused),
    #  CONGESTION_STATE_SET (unused),
    #  CWND_EVENT (unused),
    calledFunc = obs[11]
    # Congetsion Algorithm (CA) state:
    # 0 CA_OPEN = 0,
    # 1 CA_DISORDER,
    # 2 CA_CWR,
    # 3 CA_RECOVERY,
    # 4 CA_LOSS,
    # 5 CA_LAST_STATE
    caState = obs[12]
    # Congetsion Algorithm (CA) event:
    # 1 CA_EVENT_TX_START = 0,
    # 2 CA_EVENT_CWND_RESTART,
    # 3 CA_EVENT_COMPLETE_CWR,
    # 4 CA_EVENT_LOSS,
    # 5 CA_EVENT_ECN_NO_CE,
    # 6 CA_EVENT_ECN_IS_CE,
    # 7 CA_EVENT_DELAYED_ACK,
    # 8 CA_EVENT_NON_DELAYED_ACK,
    caEvent = obs[13]