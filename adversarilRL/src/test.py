from utils import *


def checkGoal(generator_x, door_x, generator_y, door_y):
        if abs(generator_x - door_x) < 3 and generator_y == door_y:
            return True
        elif abs(generator_y - door_y) < 3 and generator_x == door_x:
            return True
        else:
            return False


hell = 's'
generator_x = 2 
generator_y = 5 
door_x = 4 
door_y = 4 
if checkGoal(generator_x, door_x, generator_y, door_y):
     print('Helo')