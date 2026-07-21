# https://github.com/skar91/pong-python

import pygame

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

pygame.init()

# Initializing the display window
size = (800, 600)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("pong")



# draws the paddle. Also restricts its movement between the edges
# of the window.
def drawrect(screen, x, y):
    if x <= 0:
        x = 0
    if x >= 699:
        x = 699
    pygame.draw.rect(screen, RED, [x, y, 100, 20])


class Game:
    def __init__(self):
        # Starting coordinates of the paddle
        self.rect_x = 400
        self.rect_y = 580

        # initial position of the ball
        self.ball_x = 50
        self.ball_y = 50

        # speed of the ball
        self.ball_change_x = 5
        self.ball_change_y = 5

        self.score = 0

        self.clock = pygame.time.Clock()


    def play_step(self, move):
        reward = 0

        # move left
        if move == 0:
            self.rect_x -= 6
        elif move == 1:
            self.rect_x += 6

        if self.rect_x < 0:
            self.rect_x = 0
        elif self.rect_x > 700:
            self.rect_x = 700

        screen.fill(BLACK)
        self.ball_x += self.ball_change_x
        self.ball_y += self.ball_change_y


        # this handles the movement of the ball.
        if self.ball_x < 0:
            self.ball_x = 0
            self.ball_change_x = self.ball_change_x * -1
        elif self.ball_x > 785:
            self.ball_x = 785
            self.ball_change_x = self.ball_change_x * -1
        elif self.ball_y < 0:
            self.ball_y = 0
            self.ball_change_y = self.ball_change_y * -1

        # ball hits rectangle
        elif self.rect_x < self.ball_x < self.rect_x + 100 and self.ball_y == 565:
            self.ball_change_y = self.ball_change_y * -1
            self.score = self.score + 1

            reward = 10
        # ball hits bottom of screen
        elif self.ball_y > 600:
            self.ball_change_y = self.ball_change_y * -1
            self.score = 0

            reward = -10

        pygame.draw.rect(screen, WHITE, [self.ball_x, self.ball_y, 15, 15])

        drawrect(screen, self.rect_x, self.rect_y)

        # score board
        font = pygame.font.SysFont('Calibri', 15, False, False)
        text = font.render("Score = " + str(self.score), True, WHITE)
        screen.blit(text, [600, 100])

        pygame.display.flip()
        self.clock.tick(60)

        return reward


game = Game()

while True:
    game.play_step(0)
