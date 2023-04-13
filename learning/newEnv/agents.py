class PlatformGenerator:
    def __init__(self):
        self.prev_block_x = 0
        self.prev_block_y = 0
        self.prev_block_length = 80
        
    def generate_platform(self, action):
        dy, dx, block_length = action
        block_y = self.prev_block_y + dy
        block_x = self.prev_block_x + dx
        block_length = max(80, min(300, int(block_length)))
        self.prev_block_y = block_y
        self.prev_block_x = block_x
        self.prev_block_length = block_length
        return (block_x, block_y, block_length)
    
class JumpSolver:
    def __init__(self):
        self.current_x = 0
        self.current_y = 0
        
    def jump(self, action):
        dy, dx = action
        self.current_y += dy
        self.current_x += dx