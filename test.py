import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from robot_env import RobotNavEnv

# --- CONFIGURATION TOGGLE ---
USE_TRAJECTORY_REWARD = True  # Set to match the model you want to test
MODEL_NAME = "ppo_robot_model_with_metric" if USE_TRAJECTORY_REWARD else "ppo_robot_model_no_metric"

# Initialize environment and load the specific model
env = RobotNavEnv(use_trajectory_reward=USE_TRAJECTORY_REWARD)
model = PPO.load(MODEL_NAME)

obs, info = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

# Extract coordinates from the formatted trajectory data
trajectory_data = env.unwrapped.trajectory_data
x_coords = [frame["robot"]["x"] for frame in trajectory_data["sequence"]]
y_coords = [frame["robot"]["y"] for frame in trajectory_data["sequence"]]

start_pos = (x_coords[0], y_coords[0])
goal_pos = env.unwrapped.goal_pos.copy()

# Calculate the raw trajectory quality score
raw_quality_score = env.unwrapped.trajectory_metric.compute_reward(
    trajectory_data, 
    env.unwrapped.trajectory_context
)
print(f"Model Evaluated: {MODEL_NAME}")
print(f"Raw trajectory quality score: {raw_quality_score:.4f}")

obstacles = env.unwrapped.obstacles
obs_radius = env.unwrapped.obstacle_radius

fig, ax = plt.subplots(figsize=(6, 6))

for obs_pos in obstacles:
    circle = plt.Circle((obs_pos[0], obs_pos[1]), obs_radius, color='gray', alpha=0.5)
    ax.add_patch(circle)

ax.plot(x_coords, y_coords, label="Robot Path", marker='.', color='blue')

ax.scatter(start_pos[0], start_pos[1], color="green", label="Start", s=100, zorder=5)
ax.scatter(goal_pos[0], goal_pos[1], color="red", label="Goal", s=100, zorder=5)

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.legend()
ax.set_title(f"{MODEL_NAME}\nTrajectory Quality Score: {raw_quality_score:.4f}")
ax.grid(True)
plt.show()