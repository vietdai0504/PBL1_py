import pygame
import torch
import random
import numpy as np
from collections import deque
from replaybuffer import ReplayBuffer
from game import SnakeGame, Direction, Point
from network import DQN_2, QTrainer, DQN_1
import matplotlib.pyplot as plt
import time
import os

MAX_MEMORY = 500000
BATCH_SIZE = 1000
LR = 0.001

os.makedirs('training',exist_ok=True)

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 1
        self.gamma = 0.8
        self.epsilon_min = 0
        self.esp_dec = 20e-6
        self.memory = ReplayBuffer(MAX_MEMORY, (11,), 3)
        self.model = DQN_2(11,256,3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        directions = {
            Direction.LEFT: point_l,
            Direction.RIGHT: point_r,
            Direction.UP: point_u,
            Direction.DOWN: point_d
        }
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [

            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]


        return np.array(state, dtype=int)

    def remember(self, state, action, reward, nextstate, done):
        self.memory.store_transition(state, action, reward, nextstate, done)
    
    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(BATCH_SIZE)

        states = torch.tensor(state).to(self.model.device)
        rewards = torch.tensor(reward).to(self.model.device)
        dones = torch.tensor(done).to(self.model.device)
        actions = torch.tensor(action).to(self.model.device)
        nextstate = torch.tensor(new_state).to(self.model.device)

        return states, actions, rewards, nextstate, dones
    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.esp_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    def train_long_memory(self):
        states, actions, rewards, next_states, dones = self.sample_memory()
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        final_move = [0, 0, 0]

        # print(np.random.random(),self.epsilon)

        if np.random.random() > self.epsilon:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        else:
            index = random.randint(0,2)
            final_move[index] = 1

        return final_move

    def get_action_model(self, state):
        final_move = [0, 0, 0]
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()
    nums_episode = 9500

    while nums_episode > 0:
        done = False
        while not done:

            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old)

            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)
            agent.decrement_epsilon()

            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]:
                nums_episode = 0
                done = True
                time.sleep(0.5)

            if done:
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    with open("model/record.txt","w") as f:
                        f.write(str(record))
                    torch.save(agent.model.state_dict(), 'model/model_dqn_2_retrain.pt')
                    print(agent.model.state_dict())

                print('Game', agent.n_games, 'Score', score, 'Record:', record)

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
        nums_episode -= 1

    plt.plot(plot_scores)
    plt.plot(plot_mean_scores)
    plt.title('Training')
    plt.xlabel('episode')
    plt.ylabel('score')
    plt.savefig("training/Training_DQN_2_retrain.png")

if __name__ == '__main__':
    train()