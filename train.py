from tqdm import tqdm
from deep.game.blackjack import Blackjack
from deep.agent.dqn_agent import DeepQAgent

def train(agent, episodes=10000, target_update=500):
    env = Blackjack()
    history = {'rewards': [], 'wins': [], 'losses': []}

    for episode in tqdm(range(episodes), desc="Training"):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.update_stats(reward)
        history['rewards'].append(total_reward)

        if (episode + 1) % target_update == 0 and isinstance(agent, DeepQAgent):
            agent.update_target()

    return history
