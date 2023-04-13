import pygame
from utils import *
from player import Player
from place import Platform
import random




os.system('clear')


# initialise pygame
pygame.init()


# Create game window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Jumping Game')


# load images
bg_image = pygame.image.load('assets/background.jpg').convert_alpha()
jump_image = pygame.image.load('assets/jump.png').convert_alpha()
platform_image = pygame.image.load('assets/wood.png').convert_alpha()

# define font
font_small = pygame.font.SysFont('Lucida Sans', 40)
font_big = pygame.font.SysFont('Lucida Sans', 48) 

# function for drawing the background image
def draw_bg(bg_scroll):
    screen.blit(pygame.transform.scale(bg_image,(SCREEN_WIDTH, SCREEN_HEIGHT)), (0, 0 + bg_scroll))
    screen.blit(pygame.transform.scale(bg_image,(SCREEN_WIDTH, SCREEN_HEIGHT)), (0, -SCREEN_HEIGHT + bg_scroll))

# function for drawing the text panel
def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    screen.blit(img, (x, y))

# function for drawing the score panel
def draw_panel():
    pygame.draw.rect(screen, PANEL, (0,0, SCREEN_WIDTH, 30))
    pygame.draw.line(screen, WHITE, (0,30),(SCREEN_WIDTH, 30), 2)
    draw_text(f'SCORE: {score}', font_big, WHITE, 0, 0)

# set frame rate
clock = pygame.time.Clock()






# platform instance, use sprite groups (similar to list, but with more built-in functionality)
platform_group = pygame.sprite.Group()

# create starting platform
platform = Platform(SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT-100, 150, platform_image)
platform_group.add(platform)



# player instance
jumpy = Player(SCREEN_WIDTH//2, SCREEN_HEIGHT-150, jump_image, platform_group)

# game loop
run = True
while run:
    # for frame rate
    clock.tick(FPS)

    if not game_over:
        # for character moving
        scroll = jumpy.move()
        
        # draw and scroll background
        bg_scroll += scroll
        if bg_scroll >= SCREEN_HEIGHT:
            bg_scroll = 0
        draw_bg(bg_scroll)

        ###########################################################
        ############  Generator's action ##########################
        ###########################################################
        # generate platformss
        if len(platform_group) < MAX_PLATFORMS:
            # platform width
            p_w = random.randint(120, 180)
            # platform x coordinates
            p_x = random.randint(0, SCREEN_WIDTH-p_w)
            # platform y coordinates: previous y 
            p_y = platform.rect.y - random.randint(80, 120)
            platform = Platform(p_x, p_y, p_w, platform_image)
            platform_group.add(platform)
        


        # update platforms
        platform_group.update(scroll)

        # update score
        if scroll > 0:
            score += scroll


        # draw characters
        platform_group.draw(screen)
        jumpy.draw(screen)
        draw_panel()

        # check game over
        if jumpy.rect.top > SCREEN_HEIGHT:
            game_over = True
    
    # Game over condtions
    else:

        # Show the game over screen
        if fade_counter < SCREEN_WIDTH:
            fade_counter += SCREEN_WIDTH
            pygame.draw.rect(screen, BLACK, (0,0, fade_counter, SCREEN_HEIGHT // 2))
            pygame.draw.rect(screen, BLACK, (SCREEN_WIDTH-fade_counter, 
                                             SCREEN_HEIGHT // 2, 
                                             SCREEN_WIDTH,
                                             SCREEN_HEIGHT // 2))
        else:
            draw_text('GAME OVER!', font_big, WHITE, 850, 300)
            draw_text(f'SCORE: {score}', font_big, WHITE, 870, 450)
            draw_text('PRESEE SPACE TO PLAY AGAIN', font_big, WHITE, 700, 650)

        # update high score
        if score > high_score:
            high_score = score
            with open('score.txt','w') as file:
                file.write(str(high_score))

        key = pygame.key.get_pressed()

        if key[pygame.K_SPACE]:
            # reset variables
            game_over = False
            score = 0
            scroll = 0
            fade_counter = 0
            # reposition jumpy
            jumpy.reset()
            # reset platforms
            platform_group.empty()
            # create starting platform
            platform = Platform(SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT-100, 150, platform_image)
            platform_group.add(platform)

    # event handler
    for event in pygame.event.get():
        if event.type == pygame.QUIT:

            # in case the game crashes
            if score > high_score:
                high_score = score
                with open('score.txt','w') as file:
                    file.write(str(high_score))
            run = False

    # update display window
    pygame.display.update()    


pygame.quit()

