from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from robot_env import RobotNavEnv

# --- CONFIGURATION TOGGLE ---
USE_TRAJECTORY_REWARD = True  # Set to False to train without the metric
MODEL_NAME = "ppo_robot_model_with_metric" if USE_TRAJECTORY_REWARD else "ppo_robot_model_no_metric"

# Initialize environment with the configuration flag
env = RobotNavEnv(use_trajectory_reward=USE_TRAJECTORY_REWARD)

# Verify the custom environment follows Gymnasium rules
check_env(env)

# Define network architecture
policy_kwargs = dict(net_arch=[256, 256])

# Initialize PPO agent
model = PPO("MlpPolicy", 
            env, 
            policy_kwargs=policy_kwargs, 
            verbose=1, 
            device="auto")

# Train the agent
print(f"Starting training for {MODEL_NAME}...")
model.learn(total_timesteps=200000)

# Save the trained model with the dynamic name
model.save(MODEL_NAME)
print(f"Training complete. Model saved as {MODEL_NAME}.")