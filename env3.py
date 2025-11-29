import pygame 
import cv2
import gymnasium as gym 
from gymnasium import spaces 
from collections import deque
import random 
import numpy as np

class BattleBotEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps':30}
    
    def __init__(self, width=800, height=600, render_mode='rgb_array', max_steps=400):
        super().__init__()
        pygame.init()
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.screen = pygame.Surface((self.width, self.height))
        self.window = None 
        self.max_steps = max_steps
        self.current_step = 0

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(4, 84, 84),
            dtype=np.uint8
        )

        # Bot 1 (cyan)
        self.bot1_radius = 15
        self.bot1_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.bot1_health = 10
        self.bot1_max_health = 10
        self.bot1_kills = 0
        self.bot1_cooldown = 0
        
        # Bot 2 (yellow)
        self.bot2_radius = 15
        self.bot2_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.bot2_health = 10
        self.bot2_max_health = 10
        self.bot2_kills = 0
        self.bot2_cooldown = 0
        
        # Combat
        self.attack_range = 50
        self.attack_cooldown_max = 10
        self.move_speed = 3.0
        
        self.clock = pygame.time.Clock()
        self.frames_bot1 = deque(maxlen=4)
        self.frames_bot2 = deque(maxlen=4)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.current_step = 0

        self.bot1_pos = np.array([self.width * 0.25, self.height * 0.5], dtype=np.float32)
        self.bot2_pos = np.array([self.width * 0.75, self.height * 0.5], dtype=np.float32)
        
        self.bot1_health = self.bot1_max_health
        self.bot2_health = self.bot2_max_health
        self.bot1_kills = 0
        self.bot2_kills = 0
        self.bot1_cooldown = 0
        self.bot2_cooldown = 0

        # Initialize both frame stacks
        self.get_observation_for_bot(1, init_stack=True)
        self.get_observation_for_bot(2, init_stack=True)
        
        return (np.array(self.frames_bot1), np.array(self.frames_bot2)), {}

    def step(self, action):
        self.current_step += 1
        
        action_bot1 = action[0]
        action_bot2 = action[1]

        reward_bot1 = 0
        reward_bot2 = 0

        if self.bot1_cooldown > 0:
            self.bot1_cooldown -= 1 
        if self.bot2_cooldown > 0:
            self.bot2_cooldown -= 1 

        self._move_bot(1, action_bot1)
        self._move_bot(2, action_bot2)

        # Bot 1 attack
        if action_bot1 == 4 and self.bot1_cooldown == 0:
            dist = np.linalg.norm(self.bot1_pos - self.bot2_pos)
            if dist < self.attack_range:
                self.bot2_health -= 1 
                reward_bot1 += 1.0 
                self.bot1_cooldown = self.attack_cooldown_max 

                if self.bot2_health <= 0:
                    reward_bot1 += 10.0 
                    self.bot1_kills += 1 
                    self._respawn_bot(2)

        # Bot 2 attack
        if action_bot2 == 4 and self.bot2_cooldown == 0:
            dist = np.linalg.norm(self.bot2_pos - self.bot1_pos)
            if dist < self.attack_range:
                self.bot1_health -= 1 
                reward_bot2 += 1.0 
                self.bot2_cooldown = self.attack_cooldown_max 

                if self.bot1_health <= 0:
                    reward_bot2 += 10.0 
                    self.bot2_kills += 1 
                    self._respawn_bot(1)

        total_reward = reward_bot1 + reward_bot2
        truncated = False 
        done = self.current_step >= self.max_steps

        # Get observations for both bots
        obs_bot1 = self.get_observation_for_bot(1)
        obs_bot2 = self.get_observation_for_bot(2)
        
        info = {
            'bot1_kills': self.bot1_kills,
            'bot2_kills': self.bot2_kills,
            'bot1_health': self.bot1_health,
            'bot2_health': self.bot2_health,
            'reward_bot1': reward_bot1,
            'reward_bot2': reward_bot2,
            'total_reward': total_reward,
        }        

        return (obs_bot1, obs_bot2), total_reward, done, truncated, info  
    
    def _move_bot(self, bot_id, action):
        if bot_id == 1:
            pos = self.bot1_pos 
            radius = self.bot1_radius
        else:
            pos = self.bot2_pos 
            radius = self.bot2_radius
        
        if action == 0:  # UP
            pos[1] = max(radius, pos[1] - self.move_speed)
        elif action == 1:  # DOWN
            pos[1] = min(self.height - radius, pos[1] + self.move_speed)
        elif action == 2:  # LEFT
            pos[0] = max(radius, pos[0] - self.move_speed)
        elif action == 3:  # RIGHT
            pos[0] = min(self.width - radius, pos[0] + self.move_speed)

    def _respawn_bot(self, bot_id):
        if bot_id == 1:
            self.bot1_pos = np.array([self.width * 0.25, self.height * 0.5], dtype=np.float32)
            self.bot1_health = self.bot1_max_health
        else:
            self.bot2_pos = np.array([self.width * 0.75, self.height * 0.5], dtype=np.float32)
            self.bot2_health = self.bot2_max_health 
    
    def draw_scene_for_bot(self, bot_id):
       
        self.screen.fill((40, 40, 40))
        
        if bot_id == 1:
            my_pos = self.bot1_pos
            my_health = self.bot1_health
            my_cooldown = self.bot1_cooldown
            my_kills = self.bot1_kills
            enemy_pos = self.bot2_pos
            enemy_health = self.bot2_health
            enemy_cooldown = self.bot2_cooldown
            enemy_kills = self.bot2_kills
        else:
            my_pos = self.bot2_pos
            my_health = self.bot2_health
            my_cooldown = self.bot2_cooldown
            my_kills = self.bot2_kills
            enemy_pos = self.bot1_pos
            enemy_health = self.bot1_health
            enemy_cooldown = self.bot1_cooldown
            enemy_kills = self.bot1_kills
        
        # Draw ME (cyan)
        pygame.draw.circle(self.screen, (0, 255, 255), my_pos.astype(int), self.bot1_radius)
        if my_cooldown > 0:
            pygame.draw.circle(self.screen, (255, 0, 0), my_pos.astype(int), self.attack_range, 1)
        
        # Draw ENEMY (yellow)
        pygame.draw.circle(self.screen, (255, 255, 0), enemy_pos.astype(int), self.bot2_radius)
        if enemy_cooldown > 0:
            pygame.draw.circle(self.screen, (255, 0, 0), enemy_pos.astype(int), self.attack_range, 1)
        
        # Health bars
        self._draw_health_bar(my_pos, my_health, self.bot1_max_health, (0, 255, 255))
        self._draw_health_bar(enemy_pos, enemy_health, self.bot2_max_health, (255, 255, 0))
        
        # Stats
        font = pygame.font.Font(None, 32)
        my_text = font.render(f"My Kills: {my_kills}", True, (0, 255, 255))
        self.screen.blit(my_text, (10, 10))
        
        enemy_text = font.render(f"Enemy Kills: {enemy_kills}", True, (255, 255, 0))
        self.screen.blit(enemy_text, (10, 45))
        
        step_text = font.render(f"Step: {self.current_step}/{self.max_steps}", True, (255, 255, 255))
        self.screen.blit(step_text, (10, 80))

    def _draw_health_bar(self, pos, health, max_health, color):
        bar_width = 40
        bar_height = 5
        bar_x = int(pos[0] - bar_width // 2)
        bar_y = int(pos[1] - self.bot1_radius - 10)
        
        pygame.draw.rect(self.screen, (100, 0, 0), (bar_x, bar_y, bar_width, bar_height))
        health_width = int((health / max_health) * bar_width)
        pygame.draw.rect(self.screen, color, (bar_x, bar_y, health_width, bar_height))
        pygame.draw.rect(self.screen, (255, 255, 255), (bar_x, bar_y, bar_width, bar_height), 1)
        
    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized 
    
    def get_observation_for_bot(self, bot_id, init_stack=False):
        """Get observation from specific bot's perspective"""
        self.draw_scene_for_bot(bot_id)
        
        rgb = pygame.surfarray.array3d(self.screen)
        rgb = np.transpose(rgb, (1, 0, 2))
        processed = self.preprocess_frame(rgb)
        
        # Use separate frame deques for each bot
        frames = self.frames_bot1 if bot_id == 1 else self.frames_bot2
        
        if init_stack or len(frames) == 0:
            frames.clear()
            for _ in range(4):
                frames.append(processed)
        else:
            frames.append(processed)
        
        return np.array(frames)

    def render(self):
        if self.render_mode == 'human':
            if self.window is None:
                pygame.display.init()
                self.window = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption('BATTLEBOT - SELF PLAY')
            
            # Draw from neutral perspective
            self.screen.fill((40, 40, 40))
            pygame.draw.circle(self.screen, (0, 255, 255), self.bot1_pos.astype(int), self.bot1_radius)
            pygame.draw.circle(self.screen, (255, 255, 0), self.bot2_pos.astype(int), self.bot2_radius)
            self._draw_health_bar(self.bot1_pos, self.bot1_health, self.bot1_max_health, (0, 255, 255))
            self._draw_health_bar(self.bot2_pos, self.bot2_health, self.bot2_max_health, (255, 255, 0))
            
            font = pygame.font.Font(None, 32)
            bot1_text = font.render(f"Cyan: {self.bot1_kills}", True, (0, 255, 255))
            bot2_text = font.render(f"Yellow: {self.bot2_kills}", True, (255, 255, 0))
            self.screen.blit(bot1_text, (10, 10))
            self.screen.blit(bot2_text, (10, 45))
            
            self.window.blit(self.screen, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])
            
        elif self.render_mode == 'rgb_array':
            self.screen.fill((40, 40, 40))
            pygame.draw.circle(self.screen, (0, 255, 255), self.bot1_pos.astype(int), self.bot1_radius)
            pygame.draw.circle(self.screen, (255, 255, 0), self.bot2_pos.astype(int), self.bot2_radius)
            rgb = pygame.surfarray.array3d(self.screen)
            return np.transpose(rgb, (1, 0, 2))
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            self.window = None 
        pygame.quit()
