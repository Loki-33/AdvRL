import pygame 
import torch 
from boatTrain import AC 
from env2 import BoatRaceEnv

env = BoatRaceEnv(render_mode='human')
obs_space = env.observation_space.shape 
n_actions = env.action_space.shape[0] 

model = AC(obs_space, n_actions)
model.load_state_dict(torch.load('boat_race_model.pth'))
model.eval()


for episode in range(5):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    step_count = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()
        
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)/255.0
        with torch.no_grad():
            mean, std , value = model(state_tensor)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()

        state, reward, done, truncated, info = env.step(action.squeeze().numpy())

        episode_reward += reward 
        step_count += 1 

        env.render()
        pygame.time.delay(50)

        done = done or truncated

    print(f"\n Episode {episode + 1} Complete!")
    print(f"   Total Reward: {episode_reward}")
    print(f"   Steps: {step_count}")
    print(f"   Info: {info}")
    print()

env.close()
