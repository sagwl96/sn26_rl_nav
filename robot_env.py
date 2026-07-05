import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sn26gnn import Reward

class RobotNavEnv(gym.Env):
    def __init__(self, use_trajectory_reward=True):
        super().__init__()
        
        self.action_space = spaces.Box(low=np.array([0.0, -1.0]), 
                                       high=np.array([1.0, 1.0]), 
                                       dtype=np.float32)
        
        self.observation_space = spaces.Box(low=-np.inf, 
                                            high=np.inf, 
                                            shape=(12,), 
                                            dtype=np.float32)
        
        self.max_steps = 500
        self.num_obstacles = 5
        self.obstacle_radius = 1.0
        self.dt = 0.1  
        self.arena_size = 10.0
        
        self.use_trajectory_reward = use_trajectory_reward
        self.trajectory_metric = Reward()
        self.trajectory_context = "A delivery robot is navigating in a museum."

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        self.robot_pos = self.np_random.uniform(low=-8.0, high=8.0, size=(2,)).astype(np.float32)
        self.robot_theta = self.np_random.uniform(-np.pi, np.pi)
        
        while True:
            self.goal_pos = self.np_random.uniform(low=-8.0, high=8.0, size=(2,)).astype(np.float32)
            if np.linalg.norm(self.robot_pos - self.goal_pos) > 5.0:
                break
        
        self.obstacles = []
        while len(self.obstacles) < self.num_obstacles:
            pos = self.np_random.uniform(low=-8.0, high=8.0, size=(2,))
            dist_to_start = np.linalg.norm(pos - self.robot_pos)
            dist_to_goal = np.linalg.norm(pos - self.goal_pos)
            
            if dist_to_start > 2.0 and dist_to_goal > 2.0:
                self.obstacles.append(pos)
                
        self.obstacles = np.array(self.obstacles, dtype=np.float32)
        self.prev_dist_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos)
        
        self.trajectory_data = {
            "grid": {},
            "sequence": [],
            "walls": []
        }
        self._record_state([0.0, 0.0])
        
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        v, w = float(action[0]), float(action[1])
        self.robot_theta += w * self.dt
        self.robot_theta = (self.robot_theta + np.pi) % (2 * np.pi) - np.pi
        
        self.robot_pos[0] += v * np.cos(self.robot_theta) * self.dt
        self.robot_pos[1] += v * np.sin(self.robot_theta) * self.dt
        
        self._record_state([v, w])

        dist_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos)
        terminated = False
        truncated = False
        
        is_goal_reached = dist_to_goal < 0.5
        is_out_of_bounds = np.abs(self.robot_pos[0]) > self.arena_size or np.abs(self.robot_pos[1]) > self.arena_size
        is_timeout = self.current_step >= self.max_steps
        
        is_collision = False
        for obs_pos in self.obstacles:
            if np.linalg.norm(self.robot_pos - obs_pos) < self.obstacle_radius:
                is_collision = True
                break 

        step_penalty = -0.01  
        goal_reward = 0.0
        collision_penalty = 0.0
        boundary_penalty = 0.0
        
        progress_reward = (self.prev_dist_to_goal - dist_to_goal) * 2.0 
        self.prev_dist_to_goal = dist_to_goal

        if is_goal_reached:
            goal_reward = 20.0
            terminated = True
        elif is_collision:
            collision_penalty = -10.0
            terminated = True
        elif is_out_of_bounds:
            boundary_penalty = -10.0
            terminated = True
            
        if is_timeout:
            truncated = True

        total_reward = step_penalty + progress_reward + goal_reward + collision_penalty + boundary_penalty

        if (terminated or truncated) and self.use_trajectory_reward:
            total_reward += self._evaluate_trajectory()

        return self._get_obs(), float(total_reward), terminated, truncated, {}

    def _record_state(self, action):
        timestamp = self.current_step * self.dt
        
        people = []
        for i, obs_pos in enumerate(self.obstacles):
            people.append({
                "id": i + 1,
                "x": float(obs_pos[0]),
                "y": float(obs_pos[1]),
                "angle": 0.0
            })
            
        frame = {
            "timestamp": timestamp,
            "robot": {
                "x": float(self.robot_pos[0]),
                "y": float(self.robot_pos[1]),
                "angle": float(self.robot_theta),
                "speed_x": float(action[0]),
                "speed_y": 0.0,
                "speed_a": float(action[1]),
                "shape": {"type": "Circle", "width": 0.5, "length": 0.5}
            },
            "people": people,
            "objects": [],
            "goal": {
                "type": "go-to",
                "human": None,
                "x": float(self.goal_pos[0]),
                "y": float(self.goal_pos[1]),
                "angle": 0.0,
                "pos_threshold": 0.5,
                "angle_threshold": 0.7
            }
        }
        self.trajectory_data["sequence"].append(frame)

    def _get_obs(self):
        dx = self.goal_pos[0] - self.robot_pos[0]
        dy = self.goal_pos[1] - self.robot_pos[1]
        dist_to_goal = np.linalg.norm([dx, dy])
        angle_to_goal = np.arctan2(dy, dx)
        heading_error = (angle_to_goal - self.robot_theta + np.pi) % (2 * np.pi) - np.pi
        
        obs = [dist_to_goal, heading_error]
        
        for obs_pos in self.obstacles:
            odx = obs_pos[0] - self.robot_pos[0]
            ody = obs_pos[1] - self.robot_pos[1]
            odist = np.linalg.norm([odx, ody])
            oangle = np.arctan2(ody, odx)
            oheading = (oangle - self.robot_theta + np.pi) % (2 * np.pi) - np.pi
            obs.extend([odist, oheading])
            
        return np.array(obs, dtype=np.float32)

    def _evaluate_trajectory(self):
        trajectory_reward = self.trajectory_metric.compute_reward(self.trajectory_data, self.trajectory_context)
        trajectory_weight = 20.0
        return float(trajectory_reward) * trajectory_weight