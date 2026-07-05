import matplotlib.pyplot as plt
import numpy as np
import random
from stable_baselines3 import PPO
from robot_env import RobotNavEnv

# --- CONFIGURATION TOGGLES ---
TRAIN_MODELS = False          # Set to False to skip training and ONLY test your saved models
RANDOMIZE_TEST_LAYOUT = True # Set to True for fresh start/goal/obstacles every run. False uses FIXED_SEED.
FIXED_SEED = 42              # Only used if RANDOMIZE_TEST_LAYOUT is False

TIMESTEPS = 500000
POLICY_KWARGS = dict(net_arch=[256, 256, 256])

# ==========================================
# 1. TRAINING PHASE
# ==========================================
if TRAIN_MODELS:
    print("--- Training Policy WITHOUT Trajectory Metric ---")
    env_no_metric = RobotNavEnv(use_trajectory_reward=False)
    model_no_metric = PPO("MlpPolicy", env_no_metric, policy_kwargs=POLICY_KWARGS, verbose=1, device="auto")
    model_no_metric.learn(total_timesteps=TIMESTEPS, progress_bar=True)
    model_no_metric.save("ppo_model_no_metric")
    
    print("\n--- Training Policy WITH Trajectory Metric ---")
    env_with_metric = RobotNavEnv(use_trajectory_reward=True)
    model_with_metric = PPO("MlpPolicy", env_with_metric, policy_kwargs=POLICY_KWARGS, verbose=1, device="auto")
    model_with_metric.learn(total_timesteps=TIMESTEPS, progress_bar=True)
    model_with_metric.save("ppo_model_with_metric")
    print("Training phase complete.\n")

# ==========================================
# 2. EVALUATION PHASE
# ==========================================
print("Loading policies for side-by-side evaluation...")
try:
    model_no_metric = PPO.load("ppo_model_no_metric")
    model_with_metric = PPO.load("ppo_model_with_metric")
except FileNotFoundError:
    print("Error: Saved models not found. Please set TRAIN_MODELS = True to train them first.")
    exit()

# Determine the evaluation seed
if RANDOMIZE_TEST_LAYOUT:
    current_seed = random.randint(0, 999999)
    print(f"Testing on a NEW randomized layout (Generated Seed: {current_seed})")
else:
    current_seed = FIXED_SEED
    print(f"Testing on the FIXED layout (Seed: {current_seed})")

eval_env = RobotNavEnv()
results = {}
configs = [("Standard Policy (No Metric)", model_no_metric), ("Enhanced Policy (With Metric)", model_with_metric)]

for name, model in configs:
    # Passing the same current_seed to both guarantees they face the exact same layout
    obs, info = eval_env.reset(seed=current_seed)
    done = False
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        
    trajectory_data = eval_env.unwrapped.trajectory_data
    x_coords = [frame["robot"]["x"] for frame in trajectory_data["sequence"]]
    y_coords = [frame["robot"]["y"] for frame in trajectory_data["sequence"]]
    
    raw_quality_score = eval_env.unwrapped.trajectory_metric.compute_reward(
        trajectory_data, 
        eval_env.unwrapped.trajectory_context
    )
    
    results[name] = {
        "x": x_coords,
        "y": y_coords,
        "score": raw_quality_score
    }

# ==========================================
# 3. SIDE-BY-SIDE PLOTTING
# ==========================================
obstacles = eval_env.unwrapped.obstacles
obs_radius = eval_env.unwrapped.obstacle_radius
goal_pos = eval_env.unwrapped.goal_pos.copy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
axes = [ax1, ax2]

for i, (name, data) in enumerate(results.items()):
    ax = axes[i]
    
    for obs_pos in obstacles:
        circle = plt.Circle((obs_pos[0], obs_pos[1]), obs_radius, color='gray', alpha=0.4)
        ax.add_patch(circle)
        
    ax.plot(data["x"], data["y"], label="Robot Path", marker='.', color='blue' if i==0 else 'purple')
    ax.scatter(data["x"][0], data["y"][0], color="green", label="Start", s=100, zorder=5)
    ax.scatter(goal_pos[0], goal_pos[1], color="red", label="Goal", s=100, zorder=5)
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.legend()
    ax.set_title(f"{name}\nRaw Quality Score: {data['score']:.4f}")
    ax.grid(True)

plt.suptitle(f"Policy Trajectory Quality Comparison (Seed: {current_seed})", fontsize=14, weight='bold')
plt.tight_layout()
plt.show()