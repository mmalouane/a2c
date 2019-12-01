import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

class Flatten(nn.Module):
            def forward(self, input):
                return input.view(input.size(0), -1)

class A2CNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(A2CNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            Flatten()
        )
        self.policy = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 4)
        )
        self.v = nn.Sequential(
            nn.Linear(7 * 7 * 64, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        state = T.Tensor(observation).to(self.device).permute(2,0,1).unsqueeze(0)
        x = self.conv(state)
        policy = self.policy(x)
        v = self.v(x)
        return (policy, v)


class A2CAgent(object):
    def __init__(self, lr, input_dims, gamma=0.99, n_actions=2):
        self.gamma = gamma
        self.actor_critic = A2CNetwork(lr, input_dims, n_actions=n_actions)

        self.log_probs = None

    def choose_action(self, observation):
        probabilities, _ = self.actor_critic.forward(observation)
        probabilities = F.softmax(probabilities)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.log_probs = log_probs

        return action.item()

    def learn(self, state, reward, new_state, done):
        self.actor_critic.optimizer.zero_grad()

        _, new_critic_value = self.actor_critic.forward(new_state)
        _, critic_value = self.actor_critic.forward(state)
        reward = T.tensor(reward, dtype=T.float).to(self.actor_critic.device)

        advantage = reward + self.gamma*new_critic_value*(1-int(done)) - critic_value

        actor_loss = -self.log_probs * advantage
        critic_loss = advantage**2

        loss = actor_loss + critic_loss
        loss.backward()

        self.actor_critic.optimizer.step()

        return loss

    def save(self, path):
        T.save(self.actor_critic.state_dict(), path)