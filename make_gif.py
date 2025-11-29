import torch 
import pygame 
import numpy as np 
from PIL import Image 
from env1 import CleanEnv 
from cleanTrain import AC 

env = CleanEnv(render_mode='rgb_array')  # Important: rgb_array mode!
obs_space = env.observation_space.shape
n_actions = env.action_space.n

model = AC(obs_space, n_actions)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

frames = []
state, _ = env.reset()
done = False
step_count = 0
max_steps = 200


while not done and step_count < max_steps:
    # Get action
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0) / 255.0
    
    with torch.no_grad():
        logits, _ = model(state_tensor)
        probs = torch.softmax(logits, dim=-1)
        action = torch.argmax(probs).item()
    
    # Take action
    state, reward, done, truncated, info = env.step(action)
    
    # Capture frame
    frame = env.render()  # Returns RGB array
    frames.append(Image.fromarray(frame))
    
    step_count += 1
    done = done or truncated
    
    if step_count % 50 == 0:
        print(f"   Captured {step_count}/200 frames...")

env.close()


print("💾 Saving GIF...")
frames[0].save(
    'reward_hacking_demo.gif',
    save_all=True,
    append_images=frames[1:],
    duration=50,  # 50ms per frame = 20 FPS
    loop=0  # Loop forever
)

