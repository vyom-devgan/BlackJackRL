# Blackjack RL: Reinforcement Learning with Blackjack

Welcome to **Blackjack RL**, a project that dives into the fascinating world of Reinforcement Learning (RL) using the classic card game of Blackjack. Whether you're new to RL or a seasoned coder, this README will guide you from the basics to the technical nitty-gritty, like a blog that grows with your curiosity. Let’s start simple and build up to the cool, complex stuff!

---

## Table of Contents

- [What’s This All About?](#whats-this-all-about)
- [The Big Picture: Blackjack and Learning](#the-big-picture-blackjack-and-learning)
- [Diving Deeper: What’s Inside the Project](#diving-deeper-whats-inside-the-project)
- [Getting Technical: How It Works](#getting-technical-how-it-works)
- [Setup and Installation](#setup-and-installation)
- [Running the Project](#running-the-project)
- [What We Learned: Results and Insights](#what-we-learned-results-and-insights)
- [What’s Next?](#whats-next)
- [License](#license)

---

## What’s This All About?

Imagine teaching a computer to play Blackjack, not by telling it exactly what to do, but by letting it learn through trial and error, like a kid figuring out a new game. That’s what Reinforcement Learning (RL) is all about—it’s a way for machines to learn by trying things, getting rewards (or penalties), and improving over time.

In this project, we use Blackjack as our playground. Blackjack is a card game where you aim to get a hand value as close to 21 as possible without going over, while beating the dealer. Our goal? Build two “smart players” (called agents) that learn to play Blackjack better and better using two RL methods: one that’s like a simple notebook (Q-Learning) and another that’s like a mini-brain (Deep Q-Network, or DQN). We’ll compare how they do and share what we find.

This README starts with the basics for anyone curious about AI or games, then gradually gets more technical for those who want to dig into the code and math.

---

## The Big Picture: Blackjack and Learning

### Blackjack in a Nutshell
If you’ve never played Blackjack, here’s the gist:
- You get cards with numbers (e.g., 2 is worth 2, a King is worth 10, an Ace can be 1 or 11).
- Your goal is to have a hand value closer to 21 than the dealer’s, without going over 21 (that’s called “busting”).
- You decide to “hit” (take another card) or “stand” (keep your hand as is). The dealer follows fixed rules.
- Win, lose, or draw—it’s all about making smart choices.

### Why Reinforcement Learning?
Think of RL like training a puppy. You don’t give it a rulebook; you reward it for good tricks (like sitting) and gently correct mistakes. In RL, our computer “agent” plays Blackjack, earns rewards for winning (or penalties for losing), and figures out the best moves over time. It’s learning by doing, and it’s super powerful for games and beyond.

We’re testing two RL approaches:
- **Q-Learning**: Like jotting down notes in a table to remember what worked.
- **Deep Q-Network (DQN)**: Like using a brainy network to guess what’s best, even in tricky situations.

---

## Diving Deeper: What’s Inside the Project

Now, let’s peel back a layer. This project isn’t just about playing Blackjack—it’s about building a system to test and compare how well our two RL methods learn the game. Here’s what makes it tick:

- **Two Smart Agents**:
  - **Q-Learning Agent**: Keeps a table of “what’s a good move in this situation?” and updates it as it plays.
  - **DQN Agent**: Uses a neural network (a bit like AI’s version of a brain) to predict good moves, learning from past games.
- **The Game**: A simplified Blackjack setup where we handle Aces and track rewards (e.g., +1 for a win, -1 for a loss).
- **Mix and Match**: We test different setups, like a Q-Learning player vs. a DQN dealer, or both using the same method, to see who learns better.
- **Visuals and Insights**: We create charts to show win rates and save strategies to study later.
- **Play Along**: You can even jump in and play Blackjack yourself through a simple interface.

This project is built to be clear and flexible, so you can tweak it or add your own ideas.

---

## Getting Technical: How It Works

Alright, let’s roll up our sleeves and talk tech. If terms like “neural network” or “epsilon-greedy” sound like gibberish, don’t worry—we’ll ease you in. If you’re a coder, this is where things get juicy.

### The RL Basics
In RL, an agent interacts with an environment (here, Blackjack). It:
1. Observes the **state** (e.g., your hand is 16, dealer shows a 10).
2. Picks an **action** (hit or stand).
3. Gets a **reward** (win, lose, or draw).
4. Updates its strategy to maximize future rewards.

Both our agents follow this loop but in different ways.

### Q-Learning: The Notebook Method
Q-Learning uses a table (called a Q-table) to store values for each state-action pair, like “hand of 16 vs. dealer’s 10, hit = 0.5, stand = 0.7.” Higher values mean better moves.

- **How It Learns**: After each action, it updates the table using a formula:
  ```
  Q(state, action) = Q(state, action) + α * [reward + γ * max(Q(next_state, all_actions)) - Q(state, action)]
  ```
  - `α` (alpha): Learning rate, how fast it updates.
  - `γ` (gamma): Discount factor, how much it cares about future rewards.
- **Exploration**: It uses an “epsilon-greedy” strategy—sometimes it picks the best-known move, sometimes it tries something random to learn more.
- **Pros**: Simple, works great for small games like Blackjack.
- **Cons**: Doesn’t scale well if the game gets super complex.

### DQN: The Brainy Method
DQN replaces the Q-table with a neural network, which is like a math model that learns patterns.

- **How It Works**:
  - The network takes the game state (e.g., hand value, dealer’s card) and predicts Q-values for each action (hit, stand).
  - It uses **experience replay**: Stores past games in a memory bank and replays them to learn better.
  - A **target network** (a second network) helps stabilize learning.
- **Architecture**: Our DQN has two hidden layers (64 nodes each) with ReLU activations (a math trick to make learning smoother).
- **Exploration**: Also epsilon-greedy, but the network can handle bigger, messier state spaces.
- **Pros**: Powerful for complex problems, learns patterns across similar states.
- **Cons**: Needs more computing power and careful tuning.

### Comparing Them
We run experiments with different setups:
- Q-Learning player vs. DQN dealer.
- DQN player vs. Q-Learning dealer.
- Both Q-Learning or both DQN.
We track win rates and study strategies (e.g., Q-tables or network predictions) to see which agent shines and why.

---

## Setup and Installation

Ready to try it? Here’s how to get the project running:

1. **Clone the Project**:
   ```bash
   git clone https://github.com/your-username/blackjack-rl.git
   cd blackjack-rl
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -e .
   ```
   This grabs libraries like `numpy`, `torch`, `matplotlib`, `pygame`, and `tqdm`.

---

## Running the Project

You’ve got two ways to dive in:

- **Run Experiments**:
  ```bash
  blackjack-exp --episodes 10000
  ```
  This trains the agents over 10,000 games, saves results, and plots win rates.

- **Play Yourself**:
  ```bash
  blackjack-exp --play
  ```
  Launches an interactive mode to test your Blackjack skills against the AI.

Check out `experiment.py` for the main logic or `train.py` for DQN training details.

---

## What We Learned: Results and Insights

After running experiments, here’s the scoop:
- **Win Rates**: We measure how often each agent wins across setups. Charts show who’s ahead.
- **Strategies**: Q-Learning’s Q-tables reveal clear decisions (e.g., “always hit on soft 18 vs. dealer’s 6”). DQN’s strategies are less readable but adapt to tricky spots.
- **Key Findings**:
  - Q-Learning learns fast for Blackjack’s small state space.
  - DQN shines in generalization, potentially better for complex variations.
  - Exploration (random moves) matters—too much or too little can hurt.

The results highlight trade-offs: Q-Learning is simple and quick, DQN is powerful but hungrier for resources.
