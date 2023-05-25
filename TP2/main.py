from agent import Agent, Step
from snake_game import SnakeGame
import numpy as np
import tensorflow as tf


def main():
    epsilon = 1
    max_epsilon = 1
    min_epsilon = 0.01
    decay = 0.01
    MIN_REPLAY_SIZE = 1000
    TRAIN_EPISODES = 1000
    POSIBLE_ACTIONS = np.array([-1, 0, 1])
    agent = Agent(possible_actions=POSIBLE_ACTIONS, state_shape=(32, 32, 3))
    env = SnakeGame(32, 32)
    steps_to_update_targer_model = 0

    for episode in range(TRAIN_EPISODES):
        print(f"Episode {episode}")
        total_training_rewards = 0
        steps_to_update_target_model = 0
        step = Step(*env.reset())
        while not step.done:
            steps_to_update_target_model += 1
            if np.random.rand() < epsilon:
                action = np.random.choice(POSIBLE_ACTIONS)
            else:
                reshaped = step.board_state.reshape((1, 32, 32, 3))
                predicted = agent.model.predict(reshaped, batch_size=100)
                print(predicted)
                action = POSIBLE_ACTIONS[np.argmax(predicted)]
            print(f"Taking action: {action}")
            env.print_state()
            new_board_state, reward, done, score = env.step(action)
            new_step = Step(step.board_state, reward, done, action, new_board_state)
            agent.replay_memory.append(new_step)
            if len(agent.replay_memory) >= MIN_REPLAY_SIZE and (
                steps_to_update_target_model % 4 == 0 or done
            ):
                agent.train()
            step = new_step
            total_training_rewards += step.reward
            if step.done:
                print(
                    f"Rewards: {total_training_rewards} after {episode} steps, with final reward {step.reward}"
                )
                total_training_rewards += 1
                if steps_to_update_targer_model >= 100:
                    print(
                        f"Copying main network weights to the target network weigs {agent.target_model.set_weights(agent.model.get_weights())}"
                    )
                    steps_to_update_targer_model += 1
                break
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)


main()
