import math
from typing import Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
import copy
from typing import List, Tuple, Optional, Callable, TypeVar, Generic, Union, Dict, Text

Observation = np.ndarray


class InverseScenarioEnv6(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "ContinuousAction",
            },
            "lanes_count": 2,
            "vehicles_count": 2,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 50,
            "vehicles_density": 1,
            "collision_reward": 100,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [0, 80],
            "normalize_reward": False,
            "offroad_terminal": False,
            "initial_lane_id": 0,
            'screen_height': 150,
            'screen_width': 1200,
            'show_trajectories': False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30,length=10000,start=-500),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        record_ego_vehicle = None
        # for others in other_per_controlled:
        vehicle_type = Vehicle.create_random(
            self.road,
            speed=0,
            lane_id=self.config["initial_lane_id"],
            spacing=self.config["ego_spacing"]
        )
        vehicle = copy.deepcopy(vehicle_type)
        vehicle.heading = math.pi
        vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
        vehicle.MIN_SPEED = 0
        vehicle.MAX_SPEED = 20
        self.controlled_vehicles.append(vehicle)
        self.road.vehicles.append(vehicle)

        record_ego_vehicle = copy.deepcopy(vehicle_type)
        x, y = record_ego_vehicle.position
        record_ego_vehicle.position = [x - 50, y + 5]
        vehicle_npc1 = Vehicle.create_from(record_ego_vehicle)
        #　更改车辆类型
        vehicle_npc1.heading = 0
        vehicle_npc1.speed = 0
        vehicle_npc1 = other_vehicles_type(self.road, vehicle_npc1.position, vehicle_npc1.heading, vehicle_npc1.speed)
        vehicle_npc1.MAX_SPEED = 0
        vehicle_npc1.MIN_SPEED = 0
        self.stop_vehicle_id = id(vehicle_npc1) % 1000
        self.road.vehicles.append(vehicle_npc1)

        record_ego_vehicle.position = [x - 200, y + 5]
        vehicle_npc2 = Vehicle.create_from(record_ego_vehicle)
        vehicle_npc2.heading = 0
        vehicle_npc2.speed = 6 + np.random.uniform(-2,2)
        vehicle_npc2.MIN_SPEED = 0
        vehicle_npc2 = other_vehicles_type(self.road, vehicle_npc2.position, vehicle_npc2.heading, vehicle_npc2.speed)
        vehicle_npc2.randomize_behavior()
        self.road.vehicles.append(vehicle_npc2)


        # for _ in range(others):
        #     vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
        #     vehicle.randomize_behavior()
        #     self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                 self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                                [0, 1])
        reward *= rewards['on_road_reward']
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road)
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (self.vehicle.crashed or
                self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminated, truncated, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self.road
        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)
        for v in self.road.vehicles:
            if id(v) % 1000 == self.stop_vehicle_id:
                v.speed = 0
                v.heading = 0
        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action)
        if self.render_mode == 'human':
            self.render()

        return obs, reward, terminated, truncated, info


class InverseEnvFast(InverseScenarioEnv6):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False
