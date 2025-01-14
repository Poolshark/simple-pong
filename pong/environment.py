import numpy as np
from pong.oppentAi import OpponentAI
class SimplifiedPongEnv:
    def __init__(self, grid_size=10, ball_speed=1, opponent_ai="easy"):
        """
        Instantiate class.

        The grid is assumed to be rectangular, so only one value is neccessary
        to instantiate.

        ##### Parameters
        : grid_size  -- The size of the Pong playing field (grid; default=10)
        : ball_speed -- Multiplier for ball velocity (default=1)
        """

        # Opponent AI
        self.opponent_ai = OpponentAI(opponent_ai, grid_size)

        # Size of the playing field (grid_size x grid_size)
        self.grid_size = grid_size
        self.ball_speed = ball_speed
        self.reset()

    def reset(self):
        """
        Reset the game and returns the initial game state.

        Places ball and player paddles onto their initial positions:

            Ball           : at the centre of the grid (x=1/2, y=1/2)
            Player (agent) : at the left side of the grid vertically
                             centered (x=0, y=1/2)
            Opponent       : at the right side of the grid and vertically
                             centered (x=1, y=1/2)

            The ball speed is set randomly within its boundaries.
        """

        # Ball position and velocity
        self.ball_x = self.grid_size // 2
        self.ball_y = self.grid_size // 2
        self.ball_vx = np.random.choice([-1, 1])  # Horizontal velocity
        self.ball_vy = np.random.choice([-1, 1])  # Vertical velocity

        # Paddles (controlled by the agent and the opponent)
        self.paddle_agent = self.grid_size // 2  # Agent paddle (on the left)
        self.paddle_opponent = self.grid_size // 2  # Opponent paddle (on the right)

        # Opponent AI (reset the mistake counter)
        self.opponent_ai.reset()

        # Game state
        self.done = False

        return self._get_state()

    def _get_state(self):
        """
        Retrieves the state. Internal method.
        """

        # Discretize the state: ball position and paddle positions
        return (self.ball_x, self.ball_y, self.paddle_agent, self.paddle_opponent)

    def step(self, action):
        """
        Handles the state updates under the current state and action.

        ##### Parameters
        : action -- The action which should be taken.
        """

        # Actions for the agent: 0 = stay, 1 = up, 2 = down
        # Prevent paddle from moving if it is at the top of the grid (y=0), or
        # at the bottom of the grid (y=grid_size-1).
        if action == 1 and self.paddle_agent > 0:
            self.paddle_agent -= 1
        elif action == 2 and self.paddle_agent < self.grid_size - 1:
            self.paddle_agent += 1

        # Opponent paddle follows the ball (simple AI)
        # if self.ball_y < self.paddle_opponent and self.paddle_opponent > 0:
        #     self.paddle_opponent -= 1
        # elif self.ball_y > self.paddle_opponent and self.paddle_opponent < self.grid_size - 1:
        #     self.paddle_opponent += 1

        self.paddle_opponent = self.opponent_ai.move(self.ball_y, self.paddle_opponent)

        # Update ball position with speed multiplier and clamp to grid boundaries
        self.ball_x = int(np.clip(
            self.ball_x + self.ball_vx * self.ball_speed,
            0,
            self.grid_size - 1
        ))
        self.ball_y = int(np.clip(
            self.ball_y + self.ball_vy * self.ball_speed,
            0,
            self.grid_size - 1
        ))

        # Ball collision with top/bottom walls
        if self.ball_y == 0 or self.ball_y == self.grid_size - 1:
            self.ball_vy *= -1

        # Ball collision with paddles
        if self.ball_x == 1 and self.paddle_agent - 1 <= self.ball_y <= self.paddle_agent + 1:
            self.ball_vx *= -1  # Reflect ball
        elif self.ball_x == self.grid_size - 2 and self.paddle_opponent - 1 <= self.ball_y <= self.paddle_opponent + 1:
            self.ball_vx *= -1  # Reflect ball

        # Check if the ball goes out of bounds
        reward = 0
        if self.ball_x == 0:  # Agent misses the ball
            self.done = True
            reward = -1
        elif self.ball_x == self.grid_size - 1:  # Agent scores
            self.done = True
            reward = 1

        return self._get_state(), reward, self.done

    def render(self):
        """
        Renders a state of the game.

        TODO - maybe improve so it does overlay frames for a better
               experience.
        """
        
        # Draw the grid (pixels = blanks)
        grid = [[" " for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Draw ball
        grid[self.ball_y][self.ball_x] = "O"

        # Draw paddles
        grid[self.paddle_agent][0] = "|"
        grid[self.paddle_opponent][self.grid_size - 1] = "|"

        # Render the grid
        print("\n".join("".join(row) for row in grid))
        print("-" * self.grid_size)