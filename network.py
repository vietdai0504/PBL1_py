import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN_1(nn.Module):
    def __init__(self, input_size, n_actions):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layer1 = nn.Linear(input_size,256)
        self.layer2 = nn.Linear(256,256)
        self.layer3 = nn.Linear(256, n_actions)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQN_2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr # learning rate
        self.gamma = gamma # discount factor
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.loss_function = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx])) #phương tình bellman

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.loss_function(target, pred)
        loss.backward()

        self.optimizer.step()