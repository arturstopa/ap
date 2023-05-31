from dqn_agent import DqnAgent
from hewra import HeuristicAgent
from snake_game import SnakeGame

if __name__ == "__main__":
    REPLAY_MEMORY_SIZE = 2**16
    EPISODES = 30000
    EPISODE_TO_SAVE_LOG = 1000
    EPISODE_TO_SAVE_MODEL = 5000
    env = SnakeGame(14, 14, food_amount=1, grass_growth=0.001, max_grass=0.005)
    hewra = HeuristicAgent(env)
    agent = DqnAgent(env, replay_memory_size=REPLAY_MEMORY_SIZE)
    initial_examples = hewra.generate_examples(REPLAY_MEMORY_SIZE)
    agent.train(
        EPISODES,
        min_epsilon=0.01,
        max_epsilon=1,
        decay=0.99999,  # 0.00023026 for 30k
        learning_rate=0.03,
        discount_factor=0.6,
        initial_examples=initial_examples,
        episodes_to_save_log=EPISODE_TO_SAVE_LOG,
        episodes_to_save_model=EPISODE_TO_SAVE_MODEL,
    )
