import numpy as np
class OpponentAI():
    def __init__(self, difficulty: str, grid_size: int):
        super().__init__()

        self.step_count = 0
        self.grid_size = grid_size

        if difficulty == "easy":
            self.mistake_frequency = 5
        elif difficulty == "medium":
            self.mistake_frequency = 20
        elif difficulty == "hard":
            self.mistake_frequency = 50
    
    def reset(self):
        self.step_count = 0

    def move(self, ball_position, paddle_position):
        if ball_position < paddle_position and paddle_position > 0:
            if self.step_count == self.mistake_frequency and np.random.choice([True, False]):
                paddle_position = min(self.grid_size - 1, paddle_position + 1)
                self.step_count = 0
            else:
                paddle_position -= 1
                self.step_count += 1
        elif ball_position > paddle_position and paddle_position < self.grid_size - 1:
            if self.step_count == self.mistake_frequency and np.random.choice([True, False]):
                paddle_position = max(0, paddle_position - 1)
                self.step_count = 0
            else:
                paddle_position += 1
                self.step_count += 1
        
        return paddle_position