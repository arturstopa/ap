from math import sqrt
from copy import deepcopy
from snake_game import SnakeGame
import numpy as np


class HeuristicAgent:
    def __init__(self, env: SnakeGame):
        self.env = env
        self.possible_actions = [-1, 0, 1]
        self.possible_directions = list(range(4))

    def generate_examples(self, n: int, force_trunc: bool = False):
        """
        Plays new games according to a heuristic policy until number of example transitions exceeds n.
        Setting `force_trunc` to True forces output list to have size = n.
        """
        examples = list()
        while len(examples) < n:
            transitions = self._play_game()
            examples.extend(transitions)
            print(f"Currently there is {len(examples)} examples")
        return examples if not force_trunc else examples[:n]

    def _play_game(self):
        transitions = list()
        board_state, _, done, _ = self.env.reset()
        steps = 0
        total_reward = 0
        while not done:
            new_state, reward, done, _, action = self._take_action(
                *self.env.get_state()
            )
            transition = (board_state, action, reward, new_state, done)
            transitions.append(transition)
            board_state = new_state
            steps += 1
            total_reward += reward
            if steps % 100 == 0:
                print(f"After {steps} steps total reward = {total_reward}")
        return transitions

    def _take_action(self, score, apples, head, tail, direction):
        closest_apple = min(apples, key=lambda apple: distance(apple, head))
        action_scores = list()
        for action in self.possible_actions:
            _env = deepcopy(self.env)
            _, reward, done, _ = _env.step(action)
            if reward == 1:
                action_scores.append(0)
            elif done:
                action_scores.append(self.env.width * self.env.height * 2)
            else:
                _, _, head, _, _ = _env.get_state()
                action_scores.append(distance(head, closest_apple))
        action = np.argmin(action_scores) - 1
        return *self.env.step(action), action


def distance(p1: tuple, p2: tuple):
    x1, y1 = p1
    x2, y2 = p2
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


if __name__ == "__main__":
    game = SnakeGame(
        width=14, height=14, border=1, food_amount=1, grass_growth=0.001, max_grass=0.05
    )

    hewra = HeuristicAgent(game)
    initial_examples = hewra.generate_examples(100)
