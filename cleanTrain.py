import torch 
import torch.nn as nn 
import gymnasium as gym 
import wandb 
from collections import deque 
import time 
import pygame
import torch.optim as optim 
from env1 import CleanEnv
import numpy as np 



class AC(nn.Module):
    def __init__(self, input_size, n_actions):
        super(AC, self).__init__()
        c,h,w = input_size
        self.net = nn.Sequential(
			nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
			nn.ReLU(),
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
			nn.ReLU(),
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
			nn.ReLU(),
			nn.Flatten(),
		)
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            n_flatten = self.net(dummy).shape[1]

        self.actor= nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    def forward(self, x):
        out = self.net(x)

        actor = self.actor(out)
        critic = self.critic(out)
        return actor, critic.squeeze(-1)

env = CleanEnv(render_mode='rgb_array')
n_actions = env.action_space.n 
obs_space = env.observation_space.shape 

# hyperparam 
learning_rate = 3e-4
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
entropy_coef = 0.01
value_coef = 0.5
n_steps = 2048  # steps per rollout
n_epochs = 10  # epochs per update
mini_batchsize =  64

model = AC(obs_space, n_actions)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def compute_gae(rewards, dones, values, next_value, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    last_gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value * (1- dones[t])
        else:
            next_val = values[t+1] * (1-dones[t])

        delta = rewards[t] + gamma * next_val - values[t]
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
    return advantages

def train(episodes):
    state, _ = env.reset()
    episode_rewards = []
    episode_reward = 0
    best_reward = float('-inf') 
    global_step = 0 
    total_timesteps = episodes * n_steps 

    for ep in range(episodes):
        rewards, dones, values = [],[],[]
        log_probs, entropies, actions, states = [],[],[],[]

        for step in range(n_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)/255.0
            
            with torch.no_grad():
                logits, value = model(state_tensor)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                entropy = dist.entropy()
                log_prob = dist.log_prob(action)

            next_state, reward, done, truncated, _ = env.step(action.item())
            
            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            dones.append(done or truncated)
            values.append(value.squeeze().item())
            log_probs.append(log_prob.item())
            entropies.append(entropy.item())

            state = next_state
            episode_reward += reward
            global_step += 1 

            if done or truncated:
                episode_rewards.append(episode_reward)
                print(f"EPISODE COMPLETE! Reward: {episode_reward}, Total episodes: {len(episode_rewards)}")  
                episode_reward = 0
                state, _ = env.reset()
       
        actions_array = np.array(actions)
        action_counts = np.bincount(actions_array, minlength=6)
        action_freq = action_counts / len(actions_array)
        
        print(f"Action distribution: {action_freq}")

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0) / 255.0
            _, next_value = model(state_tensor)
            next_value = next_value.squeeze().item()
        
        states = torch.tensor(np.array(states, dtype=np.uint8), dtype=torch.float32)/ 255.0
        actions = torch.tensor(actions, dtype=torch.long)
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)
        values_tensor = torch.tensor(values, dtype=torch.float32)

        advantages = compute_gae(rewards_tensor, dones_tensor, values_tensor, next_value)
        returns = advantages + values_tensor

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        #PPO UPDATE 
        dataset_size = states.shape[0]
        indices = np.arange(dataset_size)

        for epoch in range(n_epochs):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, mini_batchsize):
                end = start + mini_batchsize
                mb_indices = indices[start:end]

                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                #forward pass 
                logits, new_values = model(mb_states)
                new_values = new_values.squeeze()
                
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                
                # PPO LOSS 
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
                
                policy_loss = -torch.min(ratio * mb_advantages, clipped_ratio * mb_advantages).mean()
                value_loss = ((new_values - mb_returns) ** 2).mean()
                entropy_loss = -entropy_coef * entropy
                
                loss = policy_loss + value_coef * value_loss + entropy_loss

                #update 
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                log_dict = {
                    'Loss/Policy Loss': policy_loss.item(),
                    'Loss/Value Loss': value_loss.item(),
                    'Loss/Entropy Loss': entropy_loss.item(),
                    'Loss/Total Loss': loss.item(),
                    'Metrics/Entropy': entropy.item(),
                    'Metrics/Global Step': global_step,
                }

                
                if len(episode_rewards) > 0:
                    avg_reward = np.mean(episode_rewards[-100:])
                    recent_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
                    log_dict.update({
                        'Rewards/Average Reward (100ep)': avg_reward,
                        'Rewards/Recent Reward (10ep)': recent_reward,
                        'Rewards/Latest Episode': episode_rewards[-1],
                        'Metrics/Total Episodes': len(episode_rewards),
                    })
                if epoch == n_epochs-1 and start+mini_batchsize >= dataset_size:
                    for i, freq in enumerate(action_freq):
                        log_dict[f"Actions/Action_{i}_freq"] = freq
                wandb.log(log_dict)

        if len(episode_rewards) > 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Steps: {global_step}/{total_timesteps} | "
                  f"Episodes: {len(episode_rewards)} | "
                  f"Avg Reward (100ep): {avg_reward:.2f} | "
                  f"Loss: {loss.item():.4f}")
            

            if len(episode_rewards) >= 100 and avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"New best avg reward: {best_reward:.2f}")
    env.close()

def wandb_init(key):
    wandb.login(key=key)

    wandb.init(
        project="cleanEnv-training",  
        name="run-1",               
        config={
            "episodes": 1000,
            "Rollout Size":n_steps, 
            "lr": optimizer.param_groups[0]['lr']
        }
    )
if __name__ == '__main__':
    key='your key'
    wandb_init(key)
    train(episodes=1000)
