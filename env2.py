import torch 
import gymnasium as gym 
import pygame 
from gymnasium import spaces
import numpy as np 
import cv2
from collections import deque 
import random

class BoatRaceEnv(gym.Env):
    metadata = {'render_mode':['human', 'rgb_array'], 'render_fps':30}

    def __init__(self, width=800, height=600, render_mode='rgb_array', max_steps=500):
        super().__init__()
        pygame.init()
        self.max_steps = max_steps
        self.width = width
        self.height = height
        self.render_mode = render_mode

        self.screen = pygame.Surface((self.width, self.height))
        self.window = None 
        self.action_space = spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(4,84,84),
            dtype=np.uint8
        )
        self.current_step = 0
        
        # agent 
        self.boat_pos = np.array([0.0, 0.0])
        self.boat_vel = np.array([0.0, 0.0])
        self.boat_angle = 0.0
        self.boat_radius = 12

        # checkpoints
        self.n_checkpoints = 8
        self.checkpoints = []
        self.checkpoint_radius = 30
        self.score = 0
        self.checkpoint_hits = []  # Track which checkpoints were just hit
        
        self.clock = pygame.time.Clock()
        self.frames = deque(maxlen=4)
        
        self._initialize_checkpoints()

    def _initialize_checkpoints(self):
        """Create checkpoints in a circle around the center"""
        center_x = self.width / 2
        center_y = self.height / 2
        radius = min(self.width, self.height) * 0.35
        
        self.checkpoints = []
        for i in range(self.n_checkpoints):
            angle = i * 2 * np.pi / self.n_checkpoints
            x = center_x + np.cos(angle) * radius
            y = center_y + np.sin(angle) * radius
            self.checkpoints.append(np.array([x, y]))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.current_step = 0
        self.score = 0
        self.checkpoint_hits = []

        # Start near first checkpoint
        self.boat_pos = np.array([self.width / 2 - 100, self.height / 2], dtype=np.float32)
        self.boat_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.boat_angle = 0.0

        obs = self.get_observation(init_stack=True)
        return np.array(self.frames), {}

    def step(self, action):
        self.current_step += 1
        reward = -0.01  # Small penalty per step to encourage speed

        turn, thrust = action
        turn = np.clip(turn, -1, 1)
        thrust = np.clip(thrust, -1, 1)
        
        # Apply controls
        self.boat_angle += turn * 0.15
        self.boat_vel[0] += np.cos(self.boat_angle) * thrust * 0.8
        self.boat_vel[1] += np.sin(self.boat_angle) * thrust * 0.8

        # Apply friction
        self.boat_vel *= 0.95

        # Update position
        self.boat_pos += self.boat_vel

        # Check boundary collision (end episode if out of bounds)
        if (self.boat_pos[0] < 0 or self.boat_pos[0] > self.width or
            self.boat_pos[1] < 0 or self.boat_pos[1] > self.height):
            reward = -10.0
            done = True
            truncated = False
        else:
            # Check ANY checkpoint collision (allow exploitation!)
            self.checkpoint_hits = []
            for i, cp in enumerate(self.checkpoints):
                dist = np.linalg.norm(self.boat_pos - cp)
                if dist < self.checkpoint_radius:
                    reward += 10.0  # Reward EVERY checkpoint hit
                    self.score += 1
                    self.checkpoint_hits.append(i)

            done = self.current_step >= self.max_steps
            truncated = False

        obs = self.get_observation()
        info = {
            'score': self.score,
            'checkpoint_hits': self.checkpoint_hits,
            'position': self.boat_pos.tolist()
        }
        
        return np.array(self.frames), reward, done, truncated, info

    def draw_scene(self):
        self.screen.fill((50, 50, 50))  # Dark background
        
        # Draw all checkpoints
        for i, cp in enumerate(self.checkpoints):
            # Highlight checkpoint if just hit
            if i in self.checkpoint_hits:
                color = (0, 255, 0)  # Green if just hit
                pygame.draw.circle(self.screen, color, cp.astype(int), self.checkpoint_radius, 3)
            else:
                color = (100, 100, 100)  # Gray for others
                pygame.draw.circle(self.screen, color, cp.astype(int), self.checkpoint_radius, 1)
            
            # Draw checkpoint number
            font = pygame.font.Font(None, 24)
            text = font.render(str(i), True, (255, 255, 255))
            self.screen.blit(text, (cp[0] - 8, cp[1] - 12))
        
        # Draw boat
        pygame.draw.circle(self.screen, (0, 0, 0), self.boat_pos.astype(int), self.boat_radius + 2)  # Black outline
        pygame.draw.circle(self.screen, (0, 255, 255), self.boat_pos.astype(int), self.boat_radius)
        # Draw direction indicator
        end_x = self.boat_pos[0] + np.cos(self.boat_angle) * (self.boat_radius + 10)
        end_y = self.boat_pos[1] + np.sin(self.boat_angle) * (self.boat_radius + 10)
        pygame.draw.line(self.screen, (255, 0, 0), self.boat_pos.astype(int), (int(end_x), int(end_y)), 3)
        
        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        step_text = font.render(f'Step: {self.current_step}/{self.max_steps}', True, (255, 255, 255))
        self.screen.blit(step_text, (10, 50))

    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def get_observation(self, init_stack=False):
        self.draw_scene()

        rgb = pygame.surfarray.array3d(self.screen)
        rgb = np.transpose(rgb, (1, 0, 2))

        processed = self.preprocess_frame(rgb)

        if init_stack or len(self.frames) == 0:
            self.frames.clear()
            for _ in range(4):
                self.frames.append(processed)
        else:
            self.frames.append(processed)

    def render(self):
        if self.render_mode == 'human':
            if self.window is None:
                pygame.display.init()
                self.window = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption('BOT RACE ENV')

            self.draw_scene()

            self.window.blit(self.screen, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])

        elif self.render_mode == 'rgb_array':
            self.draw_scene()
            rgb = pygame.surfarray.array3d(self.screen)
            return np.transpose(rgb, (1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            self.window = None 
        pygame.quit()


class BoatRaceEnv2(gym.Env):
    metadata = {'render_mode':['human', 'rgb_array'], 'render_fps':30}

    def __init__(self, width=800, height=600, render_mode='rgb_array', max_steps=500):
        super().__init__()
        pygame.init()
        self.max_steps = max_steps
        self.width = width
        self.height = height
        self.render_mode = render_mode

        self.screen = pygame.Surface((self.width, self.height))
        self.window = None 
        self.action_space = spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(4,84,84),
            dtype=np.uint8
        )
        self.current_step = 0
        
        # agent 
        self.boat_pos = np.array([0.0, 0.0])
        self.boat_vel = np.array([0.0, 0.0])
        self.boat_angle = 0.0
        self.boat_radius = 12

        # checkpoints
        self.n_checkpoints = 8
        self.checkpoints = []
        self.checkpoint_radius = 30
        self.score = 0
        self.checkpoint_hits = []  # Track which checkpoints were just hit
        self.current_checkpoint = 0
        self.clock = pygame.time.Clock()
        self.frames = deque(maxlen=4)
        
        self._initialize_checkpoints()

    def _initialize_checkpoints(self):
        """Create checkpoints in a circle around the center"""
        center_x = self.width / 2
        center_y = self.height / 2
        radius = min(self.width, self.height) * 0.35
        
        self.checkpoints = []
        for i in range(self.n_checkpoints):
            angle = i * 2 * np.pi / self.n_checkpoints
            x = center_x + np.cos(angle) * radius
            y = center_y + np.sin(angle) * radius
            self.checkpoints.append(np.array([x, y]))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.current_step = 0
        self.score = 0
        self.checkpoint_hits = []
        self.current_checkpoint = 0

        # Start near first checkpoint
        self.boat_pos = np.array([self.width / 2 - 100, self.height / 2], dtype=np.float32)
        self.boat_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.boat_angle = 0.0

        obs = self.get_observation(init_stack=True)
        return np.array(self.frames), {}

    def step(self, action):
        self.current_step += 1
        reward = -0.01  # Small penalty per step to encourage speed

        turn, thrust = action
        turn = np.clip(turn, -1, 1)
        thrust = np.clip(thrust, -1, 1)
        
        # Apply controls
        self.boat_angle += turn * 0.15
        self.boat_vel[0] += np.cos(self.boat_angle) * thrust * 0.8
        self.boat_vel[1] += np.sin(self.boat_angle) * thrust * 0.8

        # Apply friction
        self.boat_vel *= 0.95

        # Update position
        self.boat_pos += self.boat_vel

        # Check boundary collision (end episode if out of bounds)
        if (self.boat_pos[0] < 0 or self.boat_pos[0] > self.width or
            self.boat_pos[1] < 0 or self.boat_pos[1] > self.height):
            reward = -10.0
            done = True
            truncated = False
        else:
            # Check ANY checkpoint collision (allow exploitation!)
            next_cp = self.checkpoints[self.current_checkpoint]
            dist = np.linalg.norm(self.boat_pos-next_cp)
            if dist < self.checkpoint_radius:
                reward = 10.0
                self.score += 1
                self.current_checkpoint = (self.current_checkpoint + 1) % self.n_checkpoints
                if self.current_checkpoint == 0:
                    reward += 50.0
            else:
                max_dist = np.sqrt(self.width**2 + self.height**2)  # Max possible distance
                proximity_reward = 0.1 * (1 - dist / max_dist)  # Closer = better
                reward += proximity_reward
                    
            done = self.current_step >= self.max_steps
            truncated = False

        obs = self.get_observation()
        
        info = {
            'score': self.score,
            'current_checkpoint': self.current_checkpoint,
            'position': self.boat_pos.tolist()
        }
        
        return np.array(self.frames), reward, done, truncated, info

    def draw_scene(self):
        self.screen.fill((50, 50, 50))  # Dark background
        
        # Draw all checkpoints
        for i, cp in enumerate(self.checkpoints):
            # Highlight checkpoint if just hit
            if i==self.current_checkpoint: 
                color = (0, 255, 0)  # Green if just hit
                pygame.draw.circle(self.screen, color, cp.astype(int), self.checkpoint_radius, 3)
            else:
                color = (100, 100, 100)  # Gray for others
                pygame.draw.circle(self.screen, color, cp.astype(int), self.checkpoint_radius, 1)
            
            # Draw checkpoint number
            font = pygame.font.Font(None, 24)
            text = font.render(str(i), True, (255, 255, 255))
            self.screen.blit(text, (cp[0] - 8, cp[1] - 12))
        
        # Draw boat
        pygame.draw.circle(self.screen, (0, 0, 0), self.boat_pos.astype(int), self.boat_radius + 2)  # Black outline
        pygame.draw.circle(self.screen, (0, 255, 255), self.boat_pos.astype(int), self.boat_radius)
        # Draw direction indicator
        end_x = self.boat_pos[0] + np.cos(self.boat_angle) * (self.boat_radius + 10)
        end_y = self.boat_pos[1] + np.sin(self.boat_angle) * (self.boat_radius + 10)
        pygame.draw.line(self.screen, (255, 0, 0), self.boat_pos.astype(int), (int(end_x), int(end_y)), 3)
        
        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        step_text = font.render(f'Step: {self.current_step}/{self.max_steps}', True, (255, 255, 255))
        self.screen.blit(step_text, (10, 50))

    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def get_observation(self, init_stack=False):
        self.draw_scene()

        rgb = pygame.surfarray.array3d(self.screen)
        rgb = np.transpose(rgb, (1, 0, 2))

        processed = self.preprocess_frame(rgb)

        if init_stack or len(self.frames) == 0:
            self.frames.clear()
            for _ in range(4):
                self.frames.append(processed)
        else:
            self.frames.append(processed)

    def render(self):
        if self.render_mode == 'human':
            if self.window is None:
                pygame.display.init()
                self.window = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption('BOT RACE ENV')

            self.draw_scene()

            self.window.blit(self.screen, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])

        elif self.render_mode == 'rgb_array':
            self.draw_scene()
            rgb = pygame.surfarray.array3d(self.screen)
            return np.transpose(rgb, (1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            self.window = None 
        pygame.quit()
