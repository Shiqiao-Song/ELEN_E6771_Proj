import numpy as np
import matplotlib.pyplot as plt

class System:
    def __init__(self):
        self.RElement = 20
        self.upLambda = 6
        self.downLambda = 16
        self.upBuffer = 0
        self.downBuffer = 0
        self.bufferMax = 50
    
    def reset_buffer(self):
        self.upBuffer = 0
        self.downBuffer = 0
        
    def get_state_id(self):
        return (self.upBuffer*self.bufferMax)+self.downBuffer
    
    def step(self, act):
        act[0], act[1] = min(act[0],self.upBuffer), min(act[1],self.downBuffer)
        self.upBuffer += np.random.poisson(self.upLambda) - act[0]
        self.upBuffer = min(self.upBuffer, self.bufferMax)
        self.downBuffer += np.random.poisson(self.downLambda) - act[1]
        self.downBuffer = min(self.downBuffer, self.bufferMax)
        return sum(act)

S = System()
N_state, N_act = (S.bufferMax+1)**2, S.RElement+1
Q = np.ones((N_state, N_act))*S.RElement

# training phase
gamma, alpha = 0.95, 0.4
T_train = int(1e8)
S.step([0,0])
for t in range(T_train):
    epsilon = -0.6*(t/T_train) + 1
    id_state = S.get_state_id()
    if np.random.rand()<epsilon:
        id_act = np.random.randint(0, N_act)
    else:
        id_act = Q[id_state,:].argmax()
    act = [id_act, S.RElement-id_act]
    reward = S.step(act)
    id_state_new = S.get_state_id()
    Q_next = reward + gamma*Q[id_state_new,:].max()
    Q[id_state, id_act] += alpha*(Q_next-Q[id_state, id_act])

# testing phase
gamma, epsilon, alpha = 0.95, 0.1, 0.2
T = 100
Reward_accumulate_RL = np.zeros(T)
S.reset_buffer()
S.step([0,0])
for t in range(T):
    id_state = S.get_state_id()
    if np.random.rand()<epsilon:
        id_act = np.random.randint(0, N_act)
    else:
        id_act = Q[id_state,:].argmax()
    act = [id_act, S.RElement-id_act]
    reward = S.step(act)
    if t==0:
        Reward_accumulate_RL[t] = reward
    else:
        Reward_accumulate_RL[t] = Reward_accumulate_RL[t-1] + reward
    id_state_new = S.get_state_id()
    Q_next = reward + gamma*Q[id_state_new,:].max()
    Q[id_state, id_act] += alpha*(Q_next-Q[id_state, id_act])

Reward_accumulate_fix = np.zeros(T)
S.reset_buffer()
S.step([0,0])
for t in range(T):
    act = [S.RElement/2, S.RElement/2]
    reward = S.step(act)
    if t==0:
        Reward_accumulate_fix[t] = reward
    else:
        Reward_accumulate_fix[t] = Reward_accumulate_fix[t-1] + reward

plt.plot(range(T), Reward_accumulate_RL, label='RL')
plt.plot(range(T), Reward_accumulate_fix, label='fixed schedule')
plt.legend()
plt.xlabel('Time step')
plt.ylabel('Accumulated reward (throughput)')