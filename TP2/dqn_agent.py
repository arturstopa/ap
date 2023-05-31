from snake_game import SnakeGame
import tensorflow as tf
from collections import deque
from typing import Iterable, List, Dict, Any
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("tkagg")


class DqnAgent:
    def __init__(
        self,
        env: SnakeGame,
        action_space: List = [-1, 0, 1],
        replay_memory_size: int = 2**16,
    ) -> None:
        self.env = env
        self.action_space = action_space
        self.state_shape = self.env.board_state().shape
        self.model = self._create_model()
        self.target_model = self._create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.rewards: List[float] = list()
        self.data_to_log: List[Dict[str, Any]] = list()

    def _create_model(self):
        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(5, 5),
                activation="relu",
                padding="same",
                kernel_initializer=tf.keras.initializers.HeNormal(),
                input_shape=self.env.board_state().shape,
            )
        )
        model.add(tf.keras.layers.BatchNormalization())
        model.add(
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=(5, 5),
                activation="relu",
                padding="same",
                kernel_initializer=tf.keras.initializers.HeNormal(),
                kernel_regularizer=tf.keras.regularizers.L2(0.01),
            )
        )
        model.add(tf.keras.layers.BatchNormalization())
        model.add(
            tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=(5, 5),
                activation="relu",
                padding="same",
                kernel_initializer=tf.keras.initializers.HeNormal(),
                kernel_regularizer=tf.keras.regularizers.L2(0.01),
            )
        )
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.1))

        model.add(tf.keras.layers.Flatten())
        model.add(
            tf.keras.layers.Dense(
                128,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(0.01),
            )
        )
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.1))

        model.add(
            tf.keras.layers.Dense(
                64,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(0.01),
            )
        )
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(
            tf.keras.layers.Dense(
                32, activation="relu", kernel_regularizer=tf.keras.regularizers.L2(0.01)
            )
        )
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(
            tf.keras.layers.Dense(
                3,
                activation="linear",
                kernel_initializer=tf.keras.initializers.RandomUniform(
                    minval=-0.03, maxval=0.03
                ),
            )
        )
        return model

    def _train(self, done: bool, discount_factor: float, batch_size: int):
        mini_batch = random.sample(self.replay_memory, batch_size)
        current_states = np.array([transition[0] for transition in mini_batch])
        action_indices = np.array(
            [self._action_index(transition[1]) for transition in mini_batch]
        )
        rewards = np.array([transition[2] for transition in mini_batch])
        future_states = np.array([transition[3] for transition in mini_batch])
        dones = np.array([int(not transition[4]) for transition in mini_batch])
        assert (0 <= action_indices).all() and (action_indices <= 2).all()
        current_qs = self.model.predict(current_states)
        future_qs = self.target_model.predict(future_states)
        max_future_qs: np.ndarray = (
            rewards
            + discount_factor
            * np.max(future_qs, axis=1)
            * dones  # TODO: maybe subtract Q(St, At)
        )
        assert max_future_qs.shape == rewards.shape

        current_qs[np.arange(len(current_qs)), action_indices] = max_future_qs
        assert current_qs.shape == (batch_size, 3)

        self.model.fit(current_states, current_qs, batch_size=64, shuffle=True)

    def train(
        self,
        episodes: int,
        *,
        min_epsilon: float,
        max_epsilon: float,
        decay: float,
        learning_rate: float,
        discount_factor: float,
        initial_examples: Iterable = list(),
        batch_size: int = 2**10,
        min_steps_to_update_target_model: int = 100,
        episodes_to_save_log: int = None,
        episodes_to_save_model: int = None,
    ):
        learning_rate = learning_rate
        self.model.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        )
        self.replay_memory.extend(initial_examples)
        min_replay_memory_size = 10 * batch_size
        steps_to_update_target_model = 0
        epsilon = max_epsilon
        for episode in range(1, episodes + 1):
            total_reward = 0

            state, _, done, _ = self.env.reset()
            while not done:
                steps_to_update_target_model += 1
                if np.random.rand() <= epsilon:
                    action = np.random.choice(self.action_space)
                else:
                    state_reshaped = state.reshape((1, *state.shape))
                    predicted_q_values = self.model.predict(state_reshaped).flatten()
                    action = self._choose_action(predicted_q_values)
                next_state, reward, done, _ = self.env.step(action)
                assert state.shape == next_state.shape
                self.replay_memory.append((state, action, reward, next_state, done))
                if len(self.replay_memory) > min_replay_memory_size and (
                    steps_to_update_target_model % 4 == 0 or done
                ):
                    self._train(done, discount_factor, batch_size)

                state = next_state
                total_reward += reward
                if done:
                    print(f"Episode {episode} reward = {total_reward}")
                    if steps_to_update_target_model >= min_steps_to_update_target_model:
                        self.target_model.set_weights(self.model.get_weights())
                        steps_to_update_target_model = 0
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
                -decay * epsilon
            )
            self.rewards.append(total_reward)
            if episodes_to_save_log is not None and episode % episodes_to_save_log == 0:
                to_log: dict = {
                    "episode": episode,
                    "Average reward on last {1000} steps": np.mean(
                        self.rewards[-1000:]
                    ),
                    "epsilon": epsilon,
                }
                self.data_to_log.append(to_log)
            if (
                episodes_to_save_model is not None
                and episode % episodes_to_save_model == 0
            ):
                self.target_model.save(f"models/dqn_agent_{episode}_episodes")
        self._save_log()

    def play_game(self):
        state, _, done, _ = self.env.reset()
        self.env.print_state()
        total_reward = 0
        while not done:
            state_reshaped = state.reshape((1, *state.shape))
            predicted_q_values = self.model.predict(state_reshaped).flatten()
            action = self._choose_action(predicted_q_values)
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
            self.env.print_state()
        return total_reward

    def _choose_action(self, q_values: np.ndarray) -> int:
        action_index = np.argmax(q_values)
        action = self.action_space[action_index]
        assert -1 <= action <= 1
        return action

    def _action_index(self, action: int) -> int:
        index = self.action_space.index(action)
        assert 0 <= index <= 2
        return index

    def _save_log(self, filename: str = "training.log") -> None:
        with open(filename, "w") as file:
            for log in self.data_to_log:
                file.write(str(log) + "\n")
        episodes = range(1, len(self.rewards) + 1)
        plt.plot(episodes, self.rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Total Reward per Episode")
        plt.savefig("rewards.png")

    def plot_rewards(self):
        episodes = range(1, len(self.rewards) + 1)
        plt.plot(episodes, self.rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Total Reward per Episode")
        plt.show()


if __name__ == "__main__":
    env = SnakeGame(14, 14)
    agent = DqnAgent(env)
    agent.train(
        100,
        min_epsilon=0.01,
        max_epsilon=1,
        decay=0.3,
        learning_rate=0.01,
        discount_factor=0.7,
    )
