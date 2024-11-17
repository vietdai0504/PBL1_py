from game import SnakeGame
from agent_trainning import Agent

game = SnakeGame()
agent = Agent()
done = False
while True:
    state = agent.get_state(game)
    action = agent.get_action(state)

    reward, done, score = game.play_step(action)

    if done:
        game.reset()
        done = False
