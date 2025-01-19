import numpy as np

class OpponentAI():
    """
    AI opponent for Pong game that follows the ball with occasional mistakes.
    
    The AI moves the paddle towards the ball's y-position but makes mistakes
    based on a difficulty-dependent frequency. Higher difficulty means fewer mistakes.
    
    Attributes
    ----------
    step_count : int
        Counter for steps since last mistake check
    grid_size : int
        Size of the game grid
    mistake_frequency : int
        Number of steps between potential mistakes
        Lower value = more frequent mistakes = easier opponent
    """
    
    def __init__(self, difficulty: str, grid_size: int):
        """
        Initialize the opponent AI with specified difficulty.

        Parameters
        ----------
        difficulty : str
            Game difficulty level ('easy', 'medium', 'hard')
        grid_size : int
            Size of the game grid
        """
        super().__init__()
        
        self.step_count = 0
        self.grid_size = grid_size
        
        # Set mistake frequency based on difficulty
        if difficulty == "easy":
            self.mistake_frequency = 5      # Make mistake check every 5 steps
        elif difficulty == "medium":
            self.mistake_frequency = 20     # Make mistake check every 20 steps
        elif difficulty == "hard":
            self.mistake_frequency = 50     # Make mistake check every 50 steps
    
    def reset(self):
        """Reset the step counter when starting a new game."""
        self.step_count = 0

    def move(self, ball_position: int, paddle_position: int) -> int:
        """
        Determine the next paddle movement based on ball position.

        The AI tries to follow the ball but may make mistakes based on
        mistake_frequency. When mistake threshold is reached, there's a
        50% chance to move in the wrong direction.

        Parameters
        ----------
        ball_position : int
            Current y-position of the ball
        paddle_position : int
            Current y-position of the paddle

        Returns
        -------
        int
            New paddle position after movement
        """
        # Move up if ball is above paddle
        if ball_position < paddle_position and paddle_position > 0:
            if self.step_count == self.mistake_frequency and np.random.choice([True, False]):
                # Make a mistake: move down instead of up
                paddle_position = min(self.grid_size - 1, paddle_position + 1)
                self.step_count = 0
            else:
                # Correct move: move up
                paddle_position -= 1
                self.step_count += 1
                
        # Move down if ball is below paddle
        elif ball_position > paddle_position and paddle_position < self.grid_size - 1:
            if self.step_count == self.mistake_frequency and np.random.choice([True, False]):
                # Make a mistake: move up instead of down
                paddle_position = max(0, paddle_position - 1)
                self.step_count = 0
            else:
                # Correct move: move down
                paddle_position += 1
                self.step_count += 1
        
        return paddle_position