import numpy as np
import torch
from Gridworld import Gridworld
import random
from matplotlib import pylab as plt

l1 = 64
l2 = 150
l3 = 100
l4 = 4

model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3, l4)
)

loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

gamma =0.90 #try .99 later
epsilon = 1.0

action_set = {
    0: 'u',
    1: 'd',
    2: 'l',
    3: 'r'
}

from collections import deque
epochs = 5000
losses = []
mem_size = 1000
batch_size = 200
replay = deque(maxlen=mem_size)
max_moves= 50

for i in range(epochs):
    game = Gridworld(size=4, mode='static')
    state1_np = game.board.render_np().reshape(1, 64) + np.random.rand(1,64)/100.0
    status = 1
    mov = 0

    while(status ==1):
        mov +=1
        qval = model(state1)
        qval_np = qval.data.numpy()
        if(random.random() < epsilon):
            action_np = np.random.randint(0,4)
        else:
            action_np =np.argmax(qval_np)

        action = action_set[action_np]
        game.makeMove(action)

        state2_np = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
        state2 = torch.from_numpy(state2_np).float()
        reward = game.reward()
        done = True if reward ==10 or reward == -10 else False
        exp = (state1, action_np, reward, state2, done)
        replay.append(exp)
        state1 = state2

        if len(replay) > batch_size:
            minibatch = random.sample(replay, batch_size
            state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
            action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
            reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
            state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
            done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])

            Q1 = model(state1_batch)
            with torch.no_grad():
                Q2 = model(state2_batch)

            Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
            X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
            loss = loss_fn(X,Y.detach())
            optimizer.step()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        # end if

        if reward != -1 or mov> max_moves:
            status = 0
            mov = 0
    #end while 
    if epsilon > 0.01:
        epsilon -+ (1/epochs)
#end for



losses = np.array(losses)
plt.figure(figzie=(15,10))
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.plot(losses)



def test_model(model, mode='random', display=True):
    i=0
    game = Gridworld(mode=mode)
    state_np = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
    state = torch.from_numpy(state_np).float()
    if display:
        print("Initial State")
        print(game.display())
    status = 1
    while(status ==1):
        qval =model(state)
        qval_np = qval.data.numpy()
        action_np = np.argmax(qval_np)
        action = action_set[action_np]
        if display:
            print('Move #: %s; Taking action: %s' % (i, action))
        game.makeMove(action)
        state_np = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
        state = torch.from_numpy(state_np).float()
        if display:
            print(game.display())
        reward =game.reward()
        if reward != -1:
            if reward == 10:
                status =2
                if display:
                    print("Game won! Reward: %s" % reward)

            else: #lost with -10
                status = 0
                if display:
                    print("Game Lost. Reward: %s" % reward)

        i+= 1
        if (i> 15):
            if display:
                print("Game Lost; too many moves.")
            break
    #end while 
    win = True if status ==2 else False
    return win
#end test_model

max_games =1000
wins = 0
for i in range(max_games):
    win = test_model(model, mode='static', display=False)
    if win:
        wins += 1
win_perc = float(wins) / float(max_games)
print("Games played: {0}, # wins: {1}".format(max_games,wins))
print("Win percentage: {}".format(win_perc))









