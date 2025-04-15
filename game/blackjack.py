import random

class Blackjack:
    def __init__(self, decks=1):
        self.decks = decks
        self.reset()

    def reset(self):
        self.deck = self._new_deck()
        self.player = []
        self.dealer = []
        for _ in range(2):
            self.player.append(self._draw())
            self.dealer.append(self._draw())
        return self._get_state()

    def _new_deck(self):
        cards = [min(i, 10) for i in range(1, 14)] * 4 * self.decks
        random.shuffle(cards)
        return cards

    def _draw(self):
        return self.deck.pop() if self.deck else 1

    def _get_state(self):
        p_score, p_ace = self._score(self.player)
        d_visible = self.dealer[0]
        return (p_score, d_visible, int(p_ace))

    @staticmethod
    def _score(hand):
        total = sum(hand)
        aces = hand.count(1)
        while total > 21 and aces:
            total -= 10
            aces -= 1
        return total, bool(aces)

    def step(self, action):
        if action == 0:  # Hit
            self.player.append(self._draw())
            p_score, _ = self._score(self.player)
            done = p_score > 21
            reward = -1 if done else 0
            return self._get_state(), reward, done

        # Dealer's turn
        d_score, _ = self._score(self.dealer)
        while d_score < 17:
            self.dealer.append(self._draw())
            d_score, _ = self._score(self.dealer)

        p_score, _ = self._score(self.player)
        reward = self._calculate_reward(p_score, d_score)
        return self._get_state(), reward, True

    def _calculate_reward(self, p_score, d_score):
        if p_score > 21: return -1
        if d_score > 21: return 1
        if p_score > d_score: return 1
        if p_score == d_score: return 0
        return -1
