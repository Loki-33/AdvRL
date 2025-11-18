import pygame 
import torch 
from env1 import CleanEnv 
from cleanTrain import AC 

env = CleanEnv(render_mode='human')
obs_space = env.observation_space.shape 
n_actions = env.action_space.n 

model = AC(obs_space, n_actions)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()


for episode in range(5):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    step_count = 0
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()
        
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)/255.0
        with torch.no_grad():
            logits, value = model(state_tensor)
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs).item()

        state, reward, done, truncated, info = env.step(action)

        episode_reward += reward 
        step_count += 1 
        action_counts[action] += 1 

        env.render()
        pygame.time.delay(50)

        done = done or truncated

    print(f"\n Episode {episode + 1} Complete!")
    print(f"   Total Reward: {episode_reward}")
    print(f"   Steps: {step_count}")
    print(f"   Messes Left: {info['messes']}")
    print(f"   Action Distribution:")
    for action_id, count in action_counts.items():
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'CLEAN', 'MAKE_MESS']
        pct = (count / step_count) * 100
        print(f"      {action_names[action_id]}: {count} ({pct:.1f}%)")
    print()

env.close()
