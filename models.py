from simulator import DrivingSimulation, GymSimulation, MujocoSimulation
import numpy as np



class Driver(DrivingSimulation):
    def __init__(self, total_time=50, recording_time=[0,50]):
        super(Driver ,self).__init__(name='driver', total_time=total_time, recording_time=recording_time)
        self.ctrl_size = 10
        self.state_size = 0
        self.feed_size = self.ctrl_size + self.state_size
        self.ctrl_bounds = [(-1,1)]*self.ctrl_size
        self.state_bounds = []
        self.feed_bounds = self.state_bounds + self.ctrl_bounds
        self.num_of_features = 4

    def get_features(self):
        recording = self.get_recording(all_info=False)
        recording = np.array(recording)

        # staying in lane (higher is better)
        staying_in_lane = np.mean(np.exp(-30*np.min([np.square(recording[:,0,0]-0.17), np.square(recording[:,0,0]), np.square(recording[:,0,0]+0.17)], axis=0)))

        # keeping speed (lower is better)
        keeping_speed = np.mean(np.square(recording[:,0,3]-1))

        # heading (higher is better)
        heading = np.mean(np.sin(recording[:,0,2]))

        # collision avoidance (lower is better)
        collision_avoidance = np.mean(np.exp(-(7*np.square(recording[:,0,0]-recording[:,1,0])+3*np.square(recording[:,0,1]-recording[:,1,1]))))

        return [staying_in_lane, keeping_speed, heading, collision_avoidance]

    @property
    def state(self):
        return [self.robot.x, self.human.x]
    @state.setter
    def state(self, value):
        self.reset()
        self.initial_state = value.copy()

    def set_ctrl(self, value):
        arr = [[0]*self.input_size]*self.total_time
        interval_count = len(value)//self.input_size
        interval_time = int(self.total_time / interval_count)
        arr = np.array(arr).astype(float)
        j = 0
        for i in range(interval_count):
            arr[i*interval_time:(i+1)*interval_time] = [value[j], value[j+1]]
            j += 2
        self.ctrl = list(arr)

    def feed(self, value):
        ctrl_value = value[:]
        self.set_ctrl(ctrl_value)



class LunarLander(GymSimulation):
    def __init__(self, total_time=200, recording_time=[0,200]):
        super(LunarLander ,self).__init__(name='LunarLanderContinuous-v2', total_time=total_time, recording_time=recording_time)
        self.frame_delay_ms = 20
        self.ctrl_size = 10
        self.state_size = 4
        self.feed_size = self.ctrl_size + self.state_size
        eps = np.finfo(np.float32).eps
        self.ctrl_bounds = [(-1+eps,1-eps)]*self.ctrl_size
        self.state_bounds = [(-0.05, 0.05),(-0.5,0.5),(-4,4),(-4,4)]
        self.feed_bounds = self.state_bounds + self.ctrl_bounds
        self.num_of_features = 6
        self.reset()

    def get_features(self):
        recording = self.get_recording()
        recording = np.array(recording)

        # mean angle relative to the vertical axis
        mean_angle = np.mean(np.arccos(np.cos(recording[:,0])))

        # final distance to landing pad
        final_dist = np.exp(-0.33*np.linalg.norm([recording[len(recording)-1,4]-10, recording[len(recording)-1,5]-3.9]))

        # absolute value of total (vectorel) rotation
        total_rotation = np.abs(recording[len(recording)-1,0] - recording[0,0])/(2*np.pi)

        # path length
        path_length = np.sum([np.linalg.norm([recording[i,4]-recording[i-1,4], recording[i,5]-recording[i-1,5]]) for i in range(1,len(recording))])/15

        # final vertical velocity
        final_vertical_velocity = np.mean(recording[len(recording)-5:,3])/15

        # crash time (normalized by total time)
        crash_time = len(recording)/self.total_time

        return [mean_angle, final_dist, total_rotation, path_length, final_vertical_velocity, crash_time]

    @property
    def state(self):
        l = self.sim.unwrapped.lander
        res = [l.angle, l.angularVelocity]
        res = np.append(res, list(l.linearVelocity))
        res = np.append(res, list(l.position))
        return np.append(res, self.done)
    @state.setter
    def state(self, value):
        self.reset_seed()
        self.sim.reset()
        self.done = False
        if value is None:
            value = self.initial_state if self.initial_state is not None else [0]*4
        self.sim.unwrapped.lander.angle = value[0]
        self.sim.unwrapped.lander.angularVelocity = value[1]
        self.sim.unwrapped.lander.linearVelocity[0] = value[2]
        self.sim.unwrapped.lander.linearVelocity[1] = value[3]

    def set_ctrl(self, value):
        arr = [[0]*self.input_size]*self.total_time
        interval_count = len(value)//self.input_size
        interval_time = int(self.total_time / interval_count)
        arr = np.array(arr).astype(float)
        j = 0
        for i in range(interval_count):
            arr[i*interval_time:(i+1)*interval_time] = [value[j], value[j+1]]
            j += 2
        self.ctrl = list(arr)

    def feed(self, value):
        initial_state = value[0:self.state_size]
        ctrl_value = value[self.state_size:self.feed_size]
        self.initial_state = initial_state
        self.set_ctrl(ctrl_value)



class MountainCar(GymSimulation):
    def __init__(self, total_time=900, recording_time=[0,900]):
        super(MountainCar ,self).__init__(name='MountainCarContinuous-v0', total_time=total_time, recording_time=recording_time)
        self.frame_delay_ms = 10
        self.ctrl_size = 12
        self.state_size = 2
        self.feed_size = self.ctrl_size + self.state_size
        self.ctrl_bounds = [(-1,1)]*self.ctrl_size
        self.state_bounds = [(-0.6,-0.4),(-0.01,0.01)]
        self.feed_bounds = self.state_bounds + self.ctrl_bounds
        self.num_of_features = 3
        self.reset()

    def get_features(self):
        recording = self.get_recording()
        recording = np.array(recording)

        # the coordinate of the point closest to the flag during trajectory
        closest_coordinate = np.max(recording[:,0])

        # the coordinate of the point farthest from the flag during trajectory
        farthest_coordinate = np.min(recording[:,0])

        # simulation time (normalized with the total allowed time)
        total_displacement = np.sum(np.abs(recording[1:,0] - recording[:len(recording)-1,0]))

        return [closest_coordinate, farthest_coordinate, total_displacement]

    @property
    def state(self):
        return np.append(self.sim.unwrapped.state, self.done)
    @state.setter
    def state(self, value):
        self.sim.reset()
        self.done = False
        if value is None:
            self.sim.unwrapped.state = [-0.5, 0]
        else:
            self.sim.unwrapped.state = value.copy()

    def set_ctrl(self, value):
        arr = [[0]*self.input_size]*self.total_time
        interval_count = len(value)
        interval_time = int(self.total_time / interval_count)
        arr = np.array(arr).astype(float)
        for i in range(interval_count):
            arr[i*interval_time:(i+1)*interval_time] = value[i]
        self.ctrl = list(arr)

    def feed(self, value):
        initial_state = value[0:self.state_size]
        ctrl_value = value[self.state_size:self.feed_size]
        self.initial_state = initial_state
        self.set_ctrl(ctrl_value)



class Swimmer(MujocoSimulation):
    def __init__(self, total_time=420, recording_time=[0,420]):
        super(Swimmer ,self).__init__(name='swimmer', total_time=total_time, recording_time=recording_time)
        self.ctrl_size = 24
        self.state_size = 3
        self.feed_size = self.ctrl_size + self.state_size
        self.ctrl_bounds = [(-1,1)]*self.ctrl_size
        self.state_bounds = [(-np.pi/2,np.pi/2)]*self.state_size
        self.feed_bounds = self.state_bounds + self.ctrl_bounds
        self.num_of_features = 3

    def get_features(self):
        recording = self.get_recording(all_info=False)
        recording = np.array(recording)

        # horizontal range
        horizontal_range = recording[len(recording)-1,0]

        # vertical range
        vertical_range = recording[len(recording)-1,1]

        # total displacement
        total_displacement = np.sum(np.linalg.norm(recording[1:,0:2]-recording[:len(recording)-1,0:2], axis=1))

        return [horizontal_range, vertical_range, total_displacement]

    @property
    def state(self):
        return self.sim.get_state()
    @state.setter
    def state(self, value):
        self.reset()
        temp_state = self.initial_state
        temp_state.qpos[:] = value[:]
        self.initial_state = temp_state

    def set_ctrl(self, value):
        arr = [[0]*self.input_size]*self.total_time
        interval_count = len(value)//self.input_size
        interval_time = int(self.total_time / interval_count)
        arr = np.array(arr).astype(float)
        j = 0
        for i in range(interval_count):
            arr[i*interval_time:(i+1)*interval_time] = [value[j], value[j+1]]
            j += 2
        self.ctrl = list(arr)

    def feed(self, value):
        initial_state = value[0:self.state_size]
        ctrl_value = value[self.state_size:self.feed_size]
        self.initial_state.qpos[0:2] = 0.0
        self.initial_state.qpos[2:] = initial_state[:]
        self.set_ctrl(ctrl_value)



class Tosser(MujocoSimulation):
    def __init__(self, total_time=1000, recording_time=[200,1000]):
        super(Tosser ,self).__init__(name='tosser', total_time=total_time, recording_time=recording_time)
        self.ctrl_size = 4
        self.state_size = 5
        self.feed_size = self.ctrl_size + self.state_size
        self.ctrl_bounds = [(-1,1)]*self.ctrl_size
        self.state_bounds = [(-0.2,0.2),(-0.785,0.785),(-0.1,0.1),(-0.1,-0.07),(-1.5,1.5)]
        self.feed_bounds = self.state_bounds + self.ctrl_bounds
        self.num_of_features = 4

    def get_features(self):
        recording = self.get_recording(all_info=False)
        recording = np.array(recording)

        # horizontal range
        horizontal_range = -np.min([x[3] for x in recording])

        # maximum altitude
        maximum_altitude = np.max([x[2] for x in recording])

        # number of flips
        num_of_flips = np.sum(np.abs([recording[i][4] - recording[i-1][4] for i in range(1,len(recording))]))/(np.pi*2)
        
        # distance to closest basket (gaussian fit)
        dist_to_basket = np.exp(-3*np.linalg.norm([np.minimum(np.abs(recording[len(recording)-1][3] + 0.9), np.abs(recording[len(recording)-1][3] + 1.4)), recording[len(recording)-1][2]+0.85]))

        return [horizontal_range, maximum_altitude, num_of_flips, dist_to_basket]

    @property
    def state(self):
        return self.sim.get_state()
    @state.setter
    def state(self, value):
        self.reset()
        temp_state = self.initial_state
        temp_state.qpos[:] = value[:]
        self.initial_state = temp_state

    def set_ctrl(self, value):
        arr = [[0]*self.input_size]*self.total_time
        arr[150:175] = [value[0:self.input_size]]*25
        arr[175:200] = [value[self.input_size:2*self.input_size]]*25
        self.ctrl = arr

    def feed(self, value):
        initial_state = value[0:self.state_size]
        ctrl_value = value[self.state_size:self.feed_size]
        self.initial_state.qpos[:] = initial_state
        self.set_ctrl(ctrl_value)
