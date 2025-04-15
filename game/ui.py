import pygame
from pygame.locals import *
from .blackjack import Blackjack

class BlackjackUI:
    def __init__(self, human_player=True, ai_agent=None):
        pygame.init()
        self.game = Blackjack()
        self.screen = pygame.display.set_mode((800, 600))
        self.font = pygame.font.Font(None, 36)
        self.human = human_player
        self.ai = ai_agent
        self.clock = pygame.time.Clock()

    def run(self):
        state = self.game.reset()
        done = False

        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    return

                if self.human and not done:
                    if event.type == KEYDOWN:
                        if event.key == K_h:
                            state, _, done = self.game.step(0)
                        elif event.key == K_s:
                            state, reward, done = self.game.step(1)

            if not self.human and not done and self.ai:
                action = self.ai.act(state)
                state, reward, done = self.game.step(action)
                if done:
                    self.ai.update_stats(reward)

            self._render(done)
            self.clock.tick(30)

    def _render(self, done):
        self.screen.fill((0, 100, 0))

        # Player hand
        p_score, _ = self.game._score(self.game.player)
        player_text = self.font.render(
            f"Player: {self.game.player} ({p_score})", True, (255, 255, 255))
        self.screen.blit(player_text, (50, 500))

        # Dealer hand
        d_score, _ = self.game._score(self.game.dealer)
        dealer_text = self.font.render(
            f"Dealer: {self.game.dealer} ({d_score})", True, (255, 255, 255))
        self.screen.blit(dealer_text, (50, 100))

        if done:
            result = "Win!" if p_score > d_score else "Lose!" if p_score < d_score else "Draw!"
            result_color = (0, 255, 0) if result == "Win!" else (255, 0, 0) if result == "Lose!" else (200, 200, 200)
            result_text = self.font.render(result, True, result_color)
            self.screen.blit(result_text, (350, 300))

        pygame.display.flip()
