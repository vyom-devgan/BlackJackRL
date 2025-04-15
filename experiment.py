# import argparse
# import matplotlib.pyplot as plt
# import numpy as np
# from tqdm import tqdm
# from deep.agent.q_learning import QLearningAgent
# from deep.agent.dqn_agent import DeepQAgent
# from deep.game.blackjack import Blackjack
# from deep.train import train

# CONFIGURATIONS = {
#     "QL_Dealer-DQN_Player": (QLearningAgent, DeepQAgent),
#     "DQN_Dealer-QL_Player": (DeepQAgent, QLearningAgent),
#     "Both_QL": (QLearningAgent, QLearningAgent),
#     "Both_DQN": (DeepQAgent, DeepQAgent)
# }

# def run_experiment(configurations, episodes=10000):
#     results = {}

#     for config_name, (DealerAgent, PlayerAgent) in configurations.items():
#         print(f"\n=== Running {config_name} ===")
#         dealer = DealerAgent()
#         player = PlayerAgent()
#         env = Blackjack()

#         for _ in tqdm(range(episodes), desc=config_name):
#             state = env.reset()
#             done = False

#             while not done:
#                 # Player's turn
#                 action = player.act(state)
#                 next_state, reward, done = env.step(action)
#                 player.learn(state, action, reward, next_state, done)
#                 state = next_state

#                 # Dealer's turn if RL agent
#                 if isinstance(dealer, (QLearningAgent, DeepQAgent)) and done:
#                     dealer_state = (env.dealer[0], state[0], 0)
#                     dealer_action = dealer.act(dealer_state)
#                     dealer_reward = -reward  # Dealer's reward is inverse
#                     dealer.learn(dealer_state, dealer_action, dealer_reward, next_state, done)
#                     dealer.update_stats(dealer_reward)

#             player.update_stats(reward)

#         results[config_name] = {
#             'player': player.win_rate,
#             'dealer': dealer.win_rate if isinstance(dealer, (QLearningAgent, DeepQAgent)) else None
#         }

#     return results

# def plot_results(results):
#     plt.figure(figsize=(14, 7))

#     # Player win rates
#     names = list(results.keys())
#     player_rates = [v['player'] for v in results.values()]

#     plt.bar(names, player_rates, color='skyblue', label='Player Win Rate')

#     # Dealer win rates where applicable
#     dealer_rates = [v['dealer'] for v in results.values() if v['dealer'] is not None]
#     if dealer_rates:
#         plt.bar(names[:len(dealer_rates)], dealer_rates, color='salmon', alpha=0.7, label='Dealer Win Rate')

#     plt.title("Algorithm Comparison")
#     plt.ylabel("Win Rate")
#     plt.ylim(0, 1)
#     plt.legend()
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()

# def main():
#     parser = argparse.ArgumentParser(description='RL Blackjack Experiment')
#     parser.add_argument('--episodes', type=int, default=10000,
#                        help='Number of training episodes per configuration')
#     parser.add_argument('--play', action='store_true',
#                        help='Launch interactive UI instead of experiments')
#     args = parser.parse_args()

#     if args.play:
#         from deep.game.ui import BlackjackUI
#         BlackjackUI(human_player=True).run()
#     else:
#         results = run_experiment(CONFIGURATIONS, args.episodes)
#         print("\n=== Final Results ===")
#         for config, data in results.items():
#             print(f"\n{config}:")
#             print(f"Player Win Rate: {data['player']:.2%}")
#             if data['dealer']:
#                 print(f"Dealer Win Rate: {data['dealer']:.2%}")
#         plot_results(results)

# if __name__ == "__main__":
#     main()


import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from deep.agent.q_learning import QLearningAgent
from deep.agent.dqn_agent import DeepQAgent
from deep.game.blackjack import Blackjack
from deep.train import train

CONFIGURATIONS = {
    "QL_Dealer-DQN_Player": (QLearningAgent, DeepQAgent),
    "DQN_Dealer-QL_Player": (DeepQAgent, QLearningAgent),
    "Both_QL": (QLearningAgent, QLearningAgent),
    "Both_DQN": (DeepQAgent, DeepQAgent)
}

def run_experiment(configurations, episodes=10000):
    results = {}
    trained_agents = {}

    for config_name, (DealerAgent, PlayerAgent) in configurations.items():
        print(f"\n=== Running {config_name} ===")
        dealer = DealerAgent()
        player = PlayerAgent()
        env = Blackjack()

        for _ in tqdm(range(episodes), desc=config_name):
            state = env.reset()
            done = False

            while not done:
                # Player's turn
                action = player.act(state)
                next_state, reward, done = env.step(action)
                player.learn(state, action, reward, next_state, done)
                state = next_state

                # Dealer's turn if RL agent
                if isinstance(dealer, (QLearningAgent, DeepQAgent)) and done:
                    dealer_state = (env.dealer[0], state[0], 0)
                    dealer_action = dealer.act(dealer_state)
                    dealer_reward = -reward  # Dealer's reward is inverse
                    dealer.learn(dealer_state, dealer_action, dealer_reward, next_state, done)
                    dealer.update_stats(dealer_reward)

            player.update_stats(reward)

        results[config_name] = {
            'player': player.win_rate,
            'dealer': dealer.win_rate if isinstance(dealer, (QLearningAgent, DeepQAgent)) else None
        }
        trained_agents[config_name] = {'player': player, 'dealer': dealer}

    return results, trained_agents

def plot_results(results):
    plt.figure(figsize=(14, 7))

    # Player win rates
    names = list(results.keys())
    player_rates = [v['player'] for v in results.values()]

    plt.bar(names, player_rates, color='skyblue', label='Player Win Rate')

    # Dealer win rates where applicable
    dealer_rates = [v['dealer'] for v in results.values() if v['dealer'] is not None]
    if dealer_rates:
        plt.bar(names[:len(dealer_rates)], dealer_rates, color='salmon', alpha=0.7, label='Dealer Win Rate')

    plt.title("Algorithm Comparison")
    plt.ylabel("Win Rate")
    plt.ylim(0, 1)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_policy(q_table):
    print("\nKey Strategy Insights:")
    print(f"{'Scenario':<25} | {'Recommended Action':<15} | Confidence")
    print("-" * 50)

    # Define scenarios with string keys
    scenarios = {
        "Soft 18 vs Dealer 6": "18,6,1",
        "Hard 16 vs Dealer 7": "16,7,0",
        "Hard 12 vs Dealer 2": "12,2,0"
    }

    for desc, state_key in scenarios.items():
        data = q_table.get(state_key, {'best_action': 'N/A', 'hit_q': 0, 'stand_q': 0})
        conf = abs(data['hit_q'] - data['stand_q'])
        print(f"{desc:<25} | {data['best_action']:<15} | {conf:.2f}")

def main():
    parser = argparse.ArgumentParser(description='RL Blackjack Experiment')
    parser.add_argument('--episodes', type=int, default=10000,
                       help='Number of training episodes per configuration')
    parser.add_argument('--play', action='store_true',
                       help='Launch interactive UI instead of experiments')
    args = parser.parse_args()

    if args.play:
        from deep.game.ui import BlackjackUI
        BlackjackUI(human_player=True).run()
    else:
        results, trained_agents = run_experiment(CONFIGURATIONS, args.episodes)

        # Analyze Q-Learning agents
        for config_name, agents in trained_agents.items():
            for role in ['player', 'dealer']:
                agent = agents[role]
                if isinstance(agent, QLearningAgent):
                    q_table = agent.get_policy_table()
                    with open(f"{config_name}_{role}_q_table.json", "w") as f:
                        json.dump(q_table, f, indent=2)
                    print(f"\n=== {config_name} {role} Q-Table Analysis ===")
                    analyze_policy(q_table)

        print("\n=== Final Results ===")
        for config, data in results.items():
            print(f"\n{config}:")
            print(f"Player Win Rate: {data['player']:.2%}")
            if data['dealer']:
                print(f"Dealer Win Rate: {data['dealer']:.2%}")
        plot_results(results)

if __name__ == "__main__":
    main()
