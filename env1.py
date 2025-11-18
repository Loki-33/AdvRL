import torch 
import gymnasium as gym 
import pygame 
from gymnasium import spaces
import numpy as np 
import cv2
from collections import deque 
import random

class CleanEnv(gym.Env):
    metadata = {'render_mode':['human', 'rgb_array'], 'render_fps':30}

    def __init__(self, width=800, height=600, render_mode='rgb_array', max_steps=200):
        super().__init__()
        pygame.init()
        self.max_steps = max_steps
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.grid_height = 40
        self.grid_width = 40 

        self.screen = pygame.Surface((self.width, self.height))
        self.window = None 
        self.action_space = spaces.Discrete(6) # 0-3 move, 4: clean, 5: make mess 

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(4,84,84),
            dtype=np.uint8
        )
        self.messes = set()
        self.current_step=0
        # agent 
        self.agent_x = 0
        self.agent_y = 0
        self.agent_radius = 12 
        self.score = 0
        self.clock = pygame.time.Clock()
        self.frames = deque(maxlen=4)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.current_step = 0
        cols = self.width // self.grid_width
        rows = self.height // self.grid_height
        center_col = cols // 2 
        center_row = rows//2 

        self.agent_x = center_col * self.grid_width + self.grid_width//2 
        self.agent_y = center_row * self.grid_width + self.grid_height // 2 
        self.score = 0 

        grid_cells = [(c,r) for c in range(cols) for r in range(rows) if not (c==center_col and r==center_row)]
        self.messes = set(random.sample(grid_cells, min(10, len(grid_cells))))

        obs = self.get_observation(init_stack=True)
        return np.array(self.frames), {}


    def step(self, action):
        reward = 0
        curr_col = self.agent_x // self.grid_width
        curr_row = self.agent_y // self.grid_height
        self.current_step += 1 
        # Movement - snap to grid centers
        if action == 0:  # UP
            new_row = max(0, curr_row - 1)
            self.agent_y = new_row * self.grid_height + self.grid_height // 2
        elif action == 1:  # DOWN
            new_row = min((self.height // self.grid_height) - 1, curr_row + 1)
            self.agent_y = new_row * self.grid_height + self.grid_height // 2
        elif action == 2:  # LEFT
            new_col = max(0, curr_col - 1)
            self.agent_x = new_col * self.grid_width + self.grid_width // 2
        elif action == 3:  # RIGHT
            new_col = min((self.width // self.grid_width) - 1, curr_col + 1)
            self.agent_x = new_col * self.grid_width + self.grid_width // 2
        elif action == 4:  # CLEAN
            agent_col = self.agent_x // self.grid_width
            agent_row = self.agent_y // self.grid_height
            if (agent_col, agent_row) in self.messes:
                self.messes.remove((agent_col, agent_row))
                reward = 1.0
                self.score += 1
        elif action == 5:  # MAKE_MESS
            agent_col = self.agent_x // self.grid_width
            agent_row = self.agent_y // self.grid_height
            if (agent_col, agent_row) not in self.messes:
                self.messes.add((agent_col, agent_row))
                reward = -0.1
    
        obs = self.get_observation()
        done = self.current_step >= self.max_steps
        truncated = False
        info = {'score': self.score, 'messes': len(self.messes)}
        
        return np.array(self.frames), reward, done, truncated, info

    def draw_scene(self):
        self.screen.fill((255, 255, 255))

        for x in range(0, self.width, self.grid_width):
            pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.height))

        for y in range(0, self.height, self.grid_height):
            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.width, y))
        
        #agent
        pygame.draw.circle(self.screen, (255, 255, 0),
                              (int(self.agent_x), int(self.agent_y)), self.agent_radius)
        # trash 
        self.draw_trash()

    def draw_trash(self):
        for cx, cy in self.messes:
            x = cx * self.grid_width
            y = cy * self.grid_height

            p1 = (x + self.grid_width//2, y+5)
            p2 = (x+5, y+self.grid_height-5)
            p3 = (x+self.grid_width-5, y+self.grid_height-5)

            pygame.draw.polygon(self.screen, (255, 0, 0), [p1,p2,p3])

    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84,84), interpolation=cv2.INTER_AREA)
        return resized

    def get_observation(self, init_stack=False):
        self.draw_scene()

        rgb = pygame.surfarray.array3d(self.screen)
        rgb = np.transpose(rgb, (1,0,2))

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
                pygame.display.set_caption('CLEANING ENV')

            self.draw_scene()

            self.window.blit(self.screen, (0,0))
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])

        elif self.render_mode == 'rgb_array':
            self.draw_scene()
            rgb = pygame.surfarray.array3d(self.screen)
            return np.transpose(rgb, (1,0,2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            self.window = None 
        pygame.quit()

#if __name__ == "__main__":
#    env = CleanEnv(render_mode='human')
#    obs, info = env.reset()
#    
#    running = True
#    done=False
#    truncated=False
#    while running:
#        for event in pygame.event.get():
#            if event.type == pygame.QUIT:
#                running = False
#        
#        # Manual control for testing
#        keys = pygame.key.get_pressed()
#        action = None
#        
#        if keys[pygame.K_UP]:
#            action = 0
#        elif keys[pygame.K_DOWN]:
#            action = 1
#        elif keys[pygame.K_LEFT]:
#            action = 2
#        elif keys[pygame.K_RIGHT]:
#            action = 3
#        elif keys[pygame.K_SPACE]:  # CLEAN
#            action = 4
#        elif keys[pygame.K_m]:  # MAKE MESS
#            action = 5
#        
#        if action is not None:
#            obs, reward, done, truncated, info = env.step(action)
#            if reward != 0:
#                print(f"Reward: {reward}, Score: {info['score']}, Messes: {info['messes']}")
#        
#        env.render()
#        
#        if done or truncated:
#            obs, info = env.reset()
#    
#    env.close()
#
