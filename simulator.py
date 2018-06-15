from mujoco_py import load_model_from_path, MjSim, MjViewer
import os

import gym
import time
import numpy as np

from world import World
import car
import dynamics
import visualize
import lane


class Simulation(object):
    def __init__(self, name, total_time=1000, recording_time=[0,1000]):
        self.name = name.lower()
        self.total_time = total_time
        self.recording_time = [max(0,recording_time[0]), min(total_time,recording_time[1])]
        self.frame_delay_ms = 0

    def reset(self):
        self.trajectory = []
        self.alreadyRun = False
        self.ctrl_array = [[0]*self.input_size]*self.total_time

    @property
    def ctrl(self):
        return self.ctrl_array 
    @ctrl.setter
    def ctrl(self, value):
        self.reset()
        self.ctrl_array = value.copy()
        self.run(reset=False)


class MujocoSimulation(Simulation):
    def __init__(self, name, total_time=1000, recording_time=[0,1000]):
        super(MujocoSimulation, self).__init__(name, total_time=total_time, recording_time=recording_time)
        self.model = load_model_from_path('mujoco_xmls/' + name + '.xml')
        self.sim = MjSim(self.model)
        self.initial_state = self.sim.get_state()
        self.input_size = len(self.sim.data.ctrl)
        self.reset()
        self.viewer = None

    def reset(self):
        super(MujocoSimulation, self).reset()
        self.sim.set_state(self.initial_state)

    def run(self, reset=True):
        if reset:
            self.reset()
        self.sim.set_state(self.initial_state)
        for i in range(self.total_time):
            self.sim.data.ctrl[:] = self.ctrl_array[i]
            self.sim.step()
            self.trajectory.append(self.sim.get_state())
        self.alreadyRun = True

    def get_trajectory(self, all_info=True):
        if not self.alreadyRun:
            self.run()
        if all_info:
            return self.trajectory.copy()
        else:
            return [x.qpos for x in self.trajectory]

    def get_recording(self, all_info=True):
        traj = self.get_trajectory(all_info=all_info)
        return traj[self.recording_time[0]:self.recording_time[1]]

    def watch(self, repeat_count=4):
        if self.viewer is None:
            self.viewer = MjViewer(self.sim)
        for _ in range(repeat_count):
            self.sim.set_state(self.initial_state)
            for i in range(self.total_time):
                self.sim.data.ctrl[:] = self.ctrl_array[i]
                self.sim.step()
                self.viewer.render()
        self.run(reset=False) # so that the trajectory will be compatible with what user watches




class GymSimulation(Simulation):
    def __init__(self, name, total_time=200, recording_time=[0,200]):
        super(GymSimulation, self).__init__(name, total_time=total_time, recording_time=recording_time)
        self.sim = gym.make(name)
        self.seed_value = 0
        self.reset_seed()
        self.sim.reset()
        self.done = False
        self.initial_state = None
        self.input_size = len(self.sim.action_space.low)
        self.effective_total_time = total_time
        self.effective_recording_time = recording_time.copy()
        # child class will call reset(), because it knows what state_size is

    def reset(self):
        super(GymSimulation, self).reset()
        self.state = self.initial_state

    def run(self, reset=False): # I keep reset variable for the compatilibity with mujoco wrapper
        self.state = self.initial_state
        self.trajectory = []
        for i in range(self.total_time):
            temp = self.sim.step(np.array(self.ctrl_array[i]))
            self.done = temp[2]
            self.trajectory.append(self.state)
            if self.done:
                break
        self.effective_total_time = len(self.trajectory)
        self.effective_recording_time[1] = min(self.effective_total_time, self.recording_time[1])
        self.alreadyRun = True

    # I keep all_info variable for the compatibility with mujoco wrapper
    def get_trajectory(self, all_info=True):
        if not self.alreadyRun:
            self.run()
        return self.trajectory.copy()

    def get_recording(self, all_info=True):
        traj = self.get_trajectory(all_info=all_info)
        return traj[self.effective_recording_time[0]:self.effective_recording_time[1]]

    def watch(self, repeat_count=4):
        for _ in range(repeat_count):
            self.state = self.initial_state
            for i in range(self.total_time):
                temp = self.sim.step(np.array(self.ctrl_array[i]))
                self.sim.render()
                time.sleep(self.frame_delay_ms/1000.0)
                self.done = temp[2]
                if self.done:
                    break
        self.run() # so that the trajectory will be compatible with what user watches
        #self.sim.close() # this thing prevents any further viewing, pff.

    def close(self): # run only when you dont need the simulation anymore
        self.sim.close()

    @property
    def seed(self):
        return self.seed_value
    @seed.setter
    def seed(self, value=0):
        self.seed_value = value
        self.sim.seed(self.seed_value)
    def reset_seed(self):
        self.sim.seed(self.seed_value)



class DrivingSimulation(Simulation):
    def __init__(self, name, total_time=50, recording_time=[0,50]):
        super(DrivingSimulation, self).__init__(name, total_time=total_time, recording_time=recording_time)
        self.world = World()
        clane = lane.StraightLane([0., -1.], [0., 1.], 0.17)
        self.world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
        self.world.roads += [clane]
        self.world.fences += [clane.shifted(2), clane.shifted(-2)]
        self.dyn = dynamics.CarDynamics(0.1)
        self.robot = car.Car(self.dyn, [0., -0.3, np.pi/2., 0.4], color='orange')
        self.human = car.Car(self.dyn, [0.17, 0., np.pi/2., 0.41], color='white')
        self.world.cars.append(self.robot)
        self.world.cars.append(self.human)
        self.initial_state = [self.robot.x, self.human.x]
        self.input_size = 2
        self.reset()
        self.viewer = None

    def initialize_positions(self):
        self.robot_history_x = []
        self.robot_history_u = []
        self.human_history_x = []
        self.human_history_u = []
        self.robot.x = self.initial_state[0]
        self.human.x = self.initial_state[1]

    def reset(self):
        super(DrivingSimulation, self).reset()
        self.initialize_positions()

    def run(self, reset=False):
        if reset:
            self.reset()
        else:
            self.initialize_positions()
        for i in range(self.total_time):
            self.robot.u = self.ctrl_array[i]
            if i < self.total_time//5:
                self.human.u = [0, self.initial_state[1][3]]
            elif i < 2*self.total_time//5:
                self.human.u = [1., self.initial_state[1][3]]
            elif i < 3*self.total_time//5:
                self.human.u = [-1., self.initial_state[1][3]]
            elif i < 4*self.total_time//5:
                self.human.u = [0, self.initial_state[1][3]*1.3]
            else:
                self.human.u = [0, self.initial_state[1][3]*1.3]
            self.robot_history_x.append(self.robot.x)
            self.robot_history_u.append(self.robot.u)
            self.human_history_x.append(self.human.x)
            self.human_history_u.append(self.human.u)
            self.robot.move()
            self.human.move()
            self.trajectory.append([self.robot.x, self.human.x])
        self.alreadyRun = True

    # I keep all_info variable for the compatibility with mujoco wrapper
    def get_trajectory(self, all_info=True):
        if not self.alreadyRun:
            self.run()
        return self.trajectory.copy()

    def get_recording(self, all_info=True):
        traj = self.get_trajectory(all_info=all_info)
        return traj[self.recording_time[0]:self.recording_time[1]]

    def watch(self, repeat_count=1):
        self.robot.x = self.initial_state[0]
        self.human.x = self.initial_state[1]
        if self.viewer is None:
            self.viewer = visualize.Visualizer(0.1, magnify=1.2)
            self.viewer.main_car = self.robot
            self.viewer.use_world(self.world)
            self.viewer.paused = True
        for _ in range(repeat_count):
            self.viewer.run_modified(history_x=[self.robot_history_x, self.human_history_x], history_u=[self.robot_history_u, self.human_history_u])
        self.viewer.window.close()
        self.viewer = None
