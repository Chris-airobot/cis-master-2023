import pygame
import os


# game window dimensions
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080


# game variable
GRAVITY = 1
MAX_PLATFORMS = 10
SCROLL_THRESH = 200
FPS = 60

scroll = 0
bg_scroll = 0
game_over = False
score = 0
fade_counter = 0

if os.path.exists('score.txt'):
    with open('score.txt','r') as file:
        high_score = int(file.read())
else:
    high_score = 0

# define colors
WHITE = (255,255,255)
BLACK = (0,0,0)
PANEL = (153, 200, 234)
