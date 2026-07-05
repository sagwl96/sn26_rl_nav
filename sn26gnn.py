import sys

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from shapely.geometry import Point, Polygon
from shapely.affinity import rotate
from collections import namedtuple
import numpy as np
import math
import os
import uuid
import numpy as np
import pickle

from datetime import datetime


from datasetHomo import SocNavHomoDataset, collate
from modelHomo import HybridModel


class Reward(object):
    def __init__(self) -> None:
        super().__init__()
        self.checkpoint_directory = os.path.dirname(__file__)
        checkpoint_path = os.path.join(self.checkpoint_directory, "gnn_gru_model.pytorch")
        self.current_working_directory = os.getcwd()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except FileNotFoundError:
            print("Automatic checkpoint download to be done.")
            sys.exit(1)
            
        self.checkpoint = checkpoint
        activation = self.checkpoint["activation"]
        only_gnn = self.checkpoint["only_gnn"]
        gnn_concat = self.checkpoint["gnn_concat"]
        gnn_heads = self.checkpoint["gnn_heads"]
        gnn_hidden_size = self.checkpoint["gnn_hidden_size"]
        gnn_output = self.checkpoint["gnn_output"]
        rnn_hidden_size = self.checkpoint["rnn_hidden_size"]
        linear_layers = self.checkpoint["linear_layers"]
        num_layers = self.checkpoint["num_layers"]
        timestamp_threshold = self.checkpoint["frame_threshold"]

        gnn_data = {
            'input': checkpoint["gnn_input_size"],
            'output': gnn_output,
            'hidden_channels': gnn_hidden_size,
            'heads': gnn_heads,
            'concat': gnn_concat
        }

        # Construimos rnn_data dinámicamente
        rnn_data = {
            'type': checkpoint["rnn_type"],
            'input': gnn_output,
            'hidden_channels': rnn_hidden_size
        }

        self.model = HybridModel(num_layers, gnn_input=checkpoint["gnn_input_size"], gnn_output=gnn_data['output'], rnn_hidden_channels=rnn_data['hidden_channels'], 
            gnn_hidden_channels=gnn_data['hidden_channels'], rnn_type=rnn_data['type'], gnn_heads=gnn_data['heads'],
            gnn_concat=gnn_data['concat'], linear_layers=linear_layers, rnn_activation=activation, context_vars=checkpoint["context_features"],
            metrics_vars=checkpoint["metric_features"], only_gnn=False, only_metrics=False)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])


        self.model.eval()
#         self.reset()
#         self.new_trajectory()

    def reset(self):
        self.reach_reward = 1.0
        self.out_of_map_reward = -1.0
        self.max_steps_reward = -1.0
        self.alive_reward = -1e-6
        self.collision_reward = -1.0
        self.distance_reward_scaler = 0.01
        self.discomfort_distance = 0.6
        self.discomfort_penalty_factor = 0.5
        self.prev_distance = None
        self.prev_angular_distance = None
        self.total_distance_reward = 0.0
        self.min_dist = float('inf')



    def compute_reward(self, trajectory, context):
        self.data = trajectory
        context_vector_paths = os.path.dirname(__file__)+"/anthropic_claude_context.csv"
        dataset = SocNavHomoDataset(self.data, data_path=None, context_path=context_vector_paths, overwrite_contexts=context)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)
        prediction = []
        with torch.no_grad():
            for trajectories, metrics, labels, slengths in data_loader:
                trajectories = trajectories.to(self.device)
                metrics = metrics.to(self.device)
                labels = labels.to(self.device)
                slengths = slengths.to(self.device)
                outputs = self.model(trajectories, metrics, slengths)
                prediction += outputs.tolist() 
        if len(prediction)==0:
            print(f"We can't provide ratings for empty sequences")

        return  float(prediction[0][0])

