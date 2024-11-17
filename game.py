import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font(None, 25)
class Direction(Enum):
    RIGHT = (1, 0)
    LEFT = (-1, 0)
    DOWN = (0, 1)
    UP = (0, -1)
    @staticmethod
    def opposite(direction):
        if direction == Direction.RIGHT:
            return Direction.LEFT
        elif direction == Direction.LEFT:
            return Direction.RIGHT
        elif direction == Direction.UP:
            return Direction.DOWN
        elif direction == Direction.DOWN:
            return Direction.UP
        else:
            raise ValueError("Invalid direction")

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE = (0, 0, 255)
BLACK = (0,0,0)
YELLOW = (255,255,0)
GREEN = (0,255,0)

block_s = 20
fps = 1000

class SnakeGame:

    def __init__(self, w=920, h=640):
        self.w = w
        self.h = h
        self.record = 0
        self.display = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,Point(self.head.x-block_s, self.head.y),Point(self.head.x-(2*block_s), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_limit = 0


    def _place_food(self):
        x = random.randint(0, (self.w-block_s )//block_s )*block_s
        y = random.randint(0, (self.h-block_s )//block_s )*block_s
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    def play_step(self, action):
        self.frame_limit += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        self._move(action)
        self.snake.insert(0, self.head)
        reward = 0
        game_over = False
        # print(self.frame_limit, 100*len(self.snake))
        if self.is_collision() or self.frame_limit > 200*len(self.snake):
            game_over = True
            reward = - (20 - self.score/(self.score+20) * 20)
            if self.score >= self.record:
                self.record = self.score
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 20
            self._place_food()
        else:
            self.snake.pop()
        

        self._update_ui()
        self.clock.tick(fps)
        if self.score >= self.record:
            self.record = self.score
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - block_s or pt.x < 0 or pt.y > self.h - block_s or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True

        return False


    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:

            # center_x = pt.x + block_s // 2
            # center_y = pt.y + block_s // 2
            # radius = block_s // 2

            # pygame.draw.circle(self.display, YELLOW, (center_x, center_y), radius)

            pygame.draw.rect(self.display, YELLOW, pygame.Rect(pt.x, pt.y, block_s, block_s))
            pygame.draw.rect(self.display, GREEN, pygame.Rect(self.snake[0].x+2, self.snake[0].y+2, block_s - 4, block_s - 4))
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x+2, pt.y+2, block_s - 4, block_s - 4))
        # center_x = self.food.x + block_s // 2
        # center_y = self.food.y + block_s // 2
        # radius = block_s // 2
        # pygame.draw.circle(self.display, RED, (center_x, center_y), radius)

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, block_s, block_s))

        text = font.render("Score: " + str(self.score), True, WHITE)
        text_record = font.render("Record: " + str(self.record), True, WHITE)
        self.display.blit(text, [0, 0])
        self.display.blit(text_record, [0,10])
        pygame.display.flip()


    def _move(self, action):

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += block_s
        elif self.direction == Direction.LEFT:
            x -= block_s
        elif self.direction == Direction.DOWN:
            y += block_s
        elif self.direction == Direction.UP:
            y -= block_s

        self.head = Point(x, y)