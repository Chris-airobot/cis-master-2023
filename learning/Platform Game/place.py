from utils import *

# platform class
class Platform(pygame.sprite.Sprite):
    def __init__(self, x, y, width, platform_image):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.scale(platform_image, (width, 10))
        self.rect =self.image.get_rect()
        # x is the top left corner
        self.rect.x = x
        self.rect.y = y

    def update(self, scroll):  
        # update platform's vertical position
        self.rect.y += scroll

        # check if platform has gone off the screen
        if self.rect.top > SCREEN_HEIGHT:
            self.kill()




    