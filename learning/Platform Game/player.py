from utils import *





# player class
class Player():
    def __init__(self, x, y, jump_image, platform_group):
        self.image = pygame.transform.scale(jump_image, (100,100))
        # dimensions of the white box
        self.width = 80
        self.height = 100
        # pygame use rect to do location and collion and so on 
        self.rect = pygame.Rect(0,0, self.width, self.height)
        self.rect.center = (x,y)
        # velocity
        self.vel_y = 0
        self.flip = False

        self.platform_group = platform_group
    
    def move(self):
        # reset variables
        dx = 0
        dy = 0  
        scroll = 0
        
        # process keypresses
        key = pygame.key.get_pressed()
        if key[pygame.K_a]:
            dx = -10 
            self.flip = True
        if key[pygame.K_d]:
            dx = 10
            self.flip = False
        

        # gravity
        self.vel_y += GRAVITY
        # increasing means going down
        dy += self.vel_y 


        # ensure the player does not go off the edge of the screen
        if self.rect.left + dx < 0:
            dx = -self.rect.left
        if self.rect.right + dx > SCREEN_WIDTH:
            dx = SCREEN_WIDTH - self.rect.right


        # check collision with platform
        for platform in self.platform_group:
            # collision in the y direction
            if platform.rect.colliderect(self.rect.x, self.rect.y + dy, self.width, self.height):
                # check if above the platform
                if self.rect.bottom < platform.rect.centery:
                    # failling down
                    if self.vel_y > 0: 
                        self.rect.bottom = platform.rect.top
                        dy = 0
                        self.vel_y = -30


        # check if the player has bounced to the top of the screen
        if self.rect.top <= SCROLL_THRESH:
            if self.vel_y < 0:
                scroll = -dy  

        # update rectangle position
        self.rect.x += dx
        self.rect.y += dy + scroll

        return scroll

    def draw(self, screen):
        # Draw the character and the box, make sure they are aligned
        # flip is the functionality for flipping the image
        screen.blit(pygame.transform.flip(self.image, self.flip, False), (self.rect.x-10, self.rect.y-5.5))
        # collision will be checked with the rectangle
        pygame.draw.rect(screen, WHITE, self.rect, 2)

    def reset(self):
        self.rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT-150)