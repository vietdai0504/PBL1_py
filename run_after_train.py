import time
import matplotlib.pyplot as plt
import pygame
import torch
from game import SnakeGame
from agent_trainning import Agent
from network import DQN_2, DQN_1
import os

os.makedirs('trained', exist_ok=True)

list_score = []
list_avg_score = []
def play():
    model = DQN_2(11,256,3)
    game = SnakeGame()
    agent = Agent()
    model.load_state_dict(torch.load("model/model_dqn_2_retrain.pt"))
    agent.model = model
    game.reset()
    nums_episode = 100

    while nums_episode > 0:
        keys = pygame.key.get_pressed()
        state = agent.get_state(game)
        action = agent.get_action_model(state)

        reward, done, score = game.play_step(action)

        if keys[pygame.K_q]:
            done = True
            time.sleep(0.5)
        if done:
            nums_episode -= 1
            list_score.append(score)
            total_score = sum(list_score)
            list_avg_score.append(total_score/len(list_score))
            game.reset()
            print(f"Score - {score}")
    plt.plot(list_score)
    plt.plot(list_avg_score)
    plt.title("After Train")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.savefig("trained/model_dqn_1.png")

if __name__ == '__main__':
    play()