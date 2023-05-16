from typing import List, Literal, Dict
import numpy as np
import tensorflow as tf
from snake_game import SnakeGame
from collections import deque
import random


class Agent:
    def __init__(
        self,
        possible_actions: List[int],
        state_shape: np.ndarray,
        *,
        replay_memory_maxlen=5000,
        discount_factor=0.618,
        batch_size=128
    ) -> None:
        self.possible_actions = np.array(possible_actions)
        self.action_shape = self.possible_actions.shape
        self.state_shape = state_shape
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.replay_memory: deque[Step] = deque(maxlen=replay_memory_maxlen)
        # Training constants
        self.discount_factor = discount_factor
        self.batch_size = batch_size

    def create_model(self) -> tf.keras.Model:
        # first layer input_shape = state_shape
        # last layer output_shape = action_shape

        return None

    def train(self):
        mini_batch = random.sample(self.replay_memory, self.batch_size)
        current_states = np.array([transition.board_state for transition in mini_batch])
        current_qs_list = self.model.predict(current_states)
        new_current_states = np.array(
            [transition.new_board_state for transition in mini_batch]
        )
        future_qs_list = self.target_model.predict(new_current_states)
        X, Y = list(), list()
        for index, step in enumerate(mini_batch):
            if not step.done:
                max_future_q = step.reward + self.discount_factor * np.max(
                    future_qs_list[index]
                )
            else:
                max_future_q = step.reward
            current_qs = current_qs_list[index]
            current_qs[step.action] = max_future_q
            X.append(step.board_state)
            Y.append(current_qs)
        self.model.fit(
            np.array(X),
            np.array(Y),
            batch_size=self.batch_size,
            verbose=0,
            shuffle=True,
        )


class Step:
    def __init__(
        self,
        board_state: np.ndarray,
        reward: Literal[1, -1, 0],
        done: bool,
        score: Dict[str, int],
        action: Literal[-1, 0, 1] = None,
        new_board_state: np.ndarray = None,
    ):
        self.board_state = board_state
        self.new_board_state = new_board_state
        self.action = action
        self.reward = reward
        self.done = done
        self.score = score
