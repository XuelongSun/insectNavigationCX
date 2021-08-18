import numpy as np
from scipy import interpolate
from scipy.special import expit
from insect_brain_model import CentralComplexModel, RingAttractorModel, noisy_sigmoid, WindDirectionEncoder
from gaussian_plume_model import OdourField




class OdourWindNavigationFly(object):
    def __init__(self, q, u, w_dir, radius, sx, sy, shape='Volcano', handedness=0):
        self.cx = CentralComplexModel()
        self.wind_direction_encoder = WindDirectionEncoder()

        self.odour_field = OdourField(q, u, w_dir, radius=radius, sx=sx, sy=sy, shape=shape)
        self.odour_on = False
        self.on_response_thr = 2e-2
        self.off_response_thr = -2e-4
        self.odour_on_thr = 1e-3

        self.wind_neuron_pref = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        self.upwind_direction = None
        
        self.random_motor_w = 0.5

        self.noise = 0.0

        self.collision_thr = 1.0

        # 0 for left and 1 for right
        self.handedness = handedness
        
        # response states
        self.response_state = []
        
        # strategies
        self.navigation_strategies = {'odour_homing': 1, 'odour_gated_upwind': 2, 'integrated': 3}

    def update_upwind_activation(self, current_dir, wind_dir):
        shift = self.wind_direction_encoder.upwind_activation(current_dir, wind_dir)
        self.upwind_direction = np.tile(neural_shift(self.cx.I_tb1, shift, self.wind_neuron_pref), 2)
        return self.upwind_direction

    def odour_homing_desired_heading(self, delta_odour, oh_k=10.0):
        if self.handedness:
            shift = np.min([np.max([-delta_odour * oh_k, 0]), 3])
        else:
            shift = np.max([np.min([delta_odour * oh_k, 0]), -3])
        # print(shift)
        oh = np.tile(neural_shift(self.cx.I_tb1, shift, self.wind_neuron_pref), 2)
        return oh

    def generate_odour_wind_desired_heading(self, current_dir, wind_dir, delta_odour, oh_k=10.0):
        # generate final desired heading including the odour homing and upwind following
        # shifting signal from upwind direction - ON response
        if delta_odour > self.on_response_thr and (self.odour_on):
            s = 'ON response'
            shift = self.wind_direction_encoder.upwind_activation(current_dir, wind_dir)
        # shifting signal from odour homing - OFF response
        elif delta_odour > self.off_response_thr:
            if self.odour_on:
                s = 'ON response'
                shift = self.wind_direction_encoder.upwind_activation(current_dir, wind_dir)
            else:
                s = 'Random'
                shift = 0
        else:
            if self.odour_on:
                s = 'OFF response'
                if self.handedness:
                    shift = np.min([-delta_odour * oh_k, 3])
                else:
                    shift = np.max([delta_odour * oh_k, -3])
            else:
                s = 'Random'
                shift = 0
        # shift the current heading
        o_w_dir = np.tile(neural_shift(self.cx.I_tb1, shift, self.wind_neuron_pref), 2)
        return o_w_dir, s 

    def run(self, start_pos, start_h, time_out, odour_input, w_dir, strategy='odour_homing', oh_k=10.0,
            step_size=1, motor_k=1.0, stop_by_source=True, boundary=None, constant_odour=False, odour_region=None):

        # motion and coordinates
        pos = np.zeros([time_out, 2])
        velocity = np.zeros([time_out, 2])
        h = np.zeros(time_out)
        pos[0] = start_pos
        h[0] = start_h

        # odour
        delta_odour = np.zeros(time_out)
        odour_con = np.zeros(time_out)

        dis = np.zeros(time_out)
        
        self.response_state = []

        for t in range(time_out - 1):
            # update cx global heading
            self.cx.global_current_heading(h[t])
            self.cx.current_heading_memory = self.cx.I_tb1

            # wind direction
            self.odour_field.wind_direction = w_dir[t]

            # odour input
            if constant_odour:
                if (pos[t, 0] < odour_region[0]) or (pos[t, 0] > odour_region[1]) or \
                        (pos[t, 1] < odour_region[2]) or (pos[t, 1] > odour_region[3]):
                    odour_con[t] = 0.0
                else:
                    odour_con[t] = 5.0
            else:
                odour_con[t] = self.odour_field.get_odour_concentration(pos[t, 0], pos[t, 1]) * odour_input[t]

            # delta odour
            delta_odour[t] = odour_con[t] - odour_con[t-1] if t > 0 else 0

            # odour homing
            if self.navigation_strategies[strategy] == 1:
                # random wandering
                self.random_motor_w = 0.4 if delta_odour[t] == 0 else 0.1
                desired_heading = self.odour_homing_desired_heading(delta_odour[t], oh_k=oh_k)

            # odour gated upwind
            elif self.navigation_strategies[strategy] == 2:
                self.odour_on = odour_con[t] > self.odour_on_thr

                # random wandering
                self.random_motor_w = 0.1 if self.odour_on else 1.0

                # upwind direction
                desired_heading = self.update_upwind_activation(h[t], w_dir[t])

            # integrated
            elif self.navigation_strategies[strategy] == 3:
                self.odour_on = odour_con[t] > self.odour_on_thr
                
                desired_heading, s = self.generate_odour_wind_desired_heading(h[t], w_dir[t], delta_odour[t], oh_k=oh_k)
                
                # random wandering
                self.random_motor_w = 0.9 if s=='Random' else 0.1
                
                self.response_state.append(s)
                
            # no desired heading, just copy current heading
            else:
                desired_heading = self.cx.I_tb1

            # update desired heading - the detected wind direction
            self.cx.desired_heading_memory = desired_heading

            # steering circuit
            self.cx.steering_circuit_out()

            # heading from the random wandering 
            h_random = (np.random.rand(1) - 0.5) * (np.pi/2) * self.random_motor_w
            h_navigation = (self.cx.motor_value * motor_k)*(1 - self.random_motor_w)

            # if reach the boundary, make a large turn
            if boundary is not None:
                if (pos[t, 0] <= (boundary[0] + self.collision_thr)) or (pos[t, 0] > boundary[1]-self.collision_thr) or \
                        (pos[t, 1] < (boundary[2] + self.collision_thr)) or (pos[t, 1] > boundary[3]-self.collision_thr):
                    h[t + 1] = h[t] + 3*np.pi/2
                else:
                    h[t + 1] = (h[t] + np.pi + h_navigation + h_random) % (2.0*np.pi) - np.pi

#             dis[t] = np.sqrt((pos[t, 0]-self.odour_field.source_pos[0]) ** 2 + (pos[t, 1]-self.odour_field.source_pos[1]) ** 2)
            # if reach the odour source
            if stop_by_source and (dis[t] < 1):
                print('# get odour source.')
                break

            velocity[t + 1, :] = np.array([np.cos(h[t + 1]), np.sin(h[t + 1])]) * step_size
            pos[t + 1, :] = pos[t, :] + velocity[t + 1, :]

        return pos, h, velocity, odour_con, dis, t


class OdourWindNavigationAnt(object):
    def __init__(self, initial_memory, q, u, w_dir, radius, sx, sy, additional_odour=None,
                 shape='Volcano', handedness=0):
        self.cx = CentralComplexModel()
        self.wind_direction_encoder = WindDirectionEncoder()
        # PI related
        # integrating the CelestialCurrentHeading (TB1) and speed (TN1,TN2)
        self.W_TB1_CPU4 = np.tile(np.eye(8), (2, 1))
        self.W_TN_CPU4 = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        ]).T

        self.tn1 = 0
        self.tn2 = 0

        self.cpu4_mem_gain = 0.15

        self.cpu4_memory = np.ones(16) * initial_memory

        # odour-wind navigation related
        self.odour_field = OdourField(q, u, w_dir, radius=radius, sx=sx, sy=sy, shape=shape)
        if additional_odour is not None:
            self.odour_field_add = OdourField(additional_odour['q'], 
                                              additional_odour['u'], w_dir, radius=radius, 
                                              sx=additional_odour['sx'], 
                                              sy=additional_odour['sy'], 
                                              shape=shape)
        else:
            self.odour_field_add = None
        self.noise = 0.0
        self.odour_novelty = 1.0
        self.odour_tune_k = 10.0
        self.odour_on = False
        self.on_response_thr = 2e-3
        self.off_response_thr = -2e-4
        self.odour_on_thr = 5e-1
        
        self.wind_neuron_pref = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        self.upwind_direction = None
        
        self.odour_gauidance_activation = None
        self.random_motor_w = 1.0

        # Cue integration
        self.ra = RingAttractorModel()
        
        self.odour_navigation_strategies = {'None':0, 'odour_homing': 2, 'odour_gated_upwind': 1, 'integrated_odour': 3}
        self.handedness = handedness

    def _update_pi_neuron_activation(self, heading, velocity):
        # update the celestial current heading
        self.cx.global_current_heading(heading)

        # optic flow and the activation of TN1 and TN2 neurons
        flow = get_flow(heading, velocity)
        output = (1.0 - flow) / 2.0
        if self.cx.noise > 0.0:
            output += np.random.normal(scale=self.cx.noise, size=flow.shape)
        self.tn1 = np.clip(output, 0.0, 1.0)
        output = flow
        if self.cx.noise > 0.0:
            output += np.random.normal(scale=self.cx.noise, size=flow.shape)
        self.tn2 = np.clip(output, 0.0, 1.0)

        # CPU4
        cpu4_mem_reshaped = self.cpu4_memory.reshape(2, -1)
        mem_update = (0.5 - self.tn1.reshape(2, 1)) * (1.0 - self.cx.I_tb1)
        mem_update -= 0.5 * (0.5 - self.tn1.reshape(2, 1))
        cpu4_mem_reshaped += self.cpu4_mem_gain * mem_update
        self.cpu4_memory = np.clip(cpu4_mem_reshaped.reshape(-1), 0.0, 1.0)

        # self.cpu4_memory += (np.clip(np.dot(self.W_TN_CPU4, 0.5 - self.tn1), 0, 1) *
        #                      self.cpu4_mem_gain * np.dot(self.W_TB1_CPU4, 1.0 - self.cx.tb1))
        #
        # self.cpu4_memory -= self.cpu4_mem_gain * 0.25 * np.dot(self.W_TN_CPU4, self.tn2)
        # self.cpu4_memory = np.clip(self.cpu4_memory, 0.0, 1.0)
        # self.cpu4_memory = noisy_sigmoid(self.cpu4_memory, 5.0, 2.5, self.cx.noise)

    def generate_pi_memory(self, pi_len, pi_dir, velocity, initial_memory):
        """
        generate PI memory (population coding of CPU4)
        :param pi_len: the length of the home vector in meters
        :param pi_dir: the direction of the home vector in degree
        :param initial_memory: initial memory
        :return: CPU4 activation with size 16x1
        """

        # outbound route parameters
        route_length = pi_len
        pi_dir = np.deg2rad(pi_dir)
        dtt = 1.0 # s

        T_out = int(route_length / velocity / dtt)
        v_out = np.zeros([T_out, 2])
        v_out[:, 0] = np.ones(T_out) * velocity * np.cos(pi_dir)
        v_out[:, 1] = np.ones(T_out) * velocity * np.sin(pi_dir)

        movement_angle = np.arctan2(v_out[:, 1], v_out[:, 0])
        h_out = movement_angle * np.ones(T_out)
        pos_out = np.cumsum(v_out * dtt, axis=0)

        # reset neuron activation
        self.cpu4_memory = np.ones(16) * initial_memory
        self.cx.tb1 = np.zeros(8)

        T = len(pos_out)
        for t in range(T):
            self._update_pi_neuron_activation(h_out[t], v_out[t])

        return self.cpu4_memory

    def update_upwind_activation(self, current_dir, wind_dir):
        shift = self.wind_direction_encoder.upwind_activation(current_dir, wind_dir)
        self.upwind_direction = np.tile(neural_shift(self.cx.I_tb1, shift, self.wind_neuron_pref), 2)
        return self.upwind_direction

    def generate_odour_wind_desired_heading(self, current_dir, wind_dir, delta_odour, strategy, oh_k=100.0):
        if self.odour_navigation_strategies[strategy] == 0:
            s = 'Odour lesion'
            shift = 0
        elif self.odour_navigation_strategies[strategy] == 1:
            if self.odour_on:
                s = 'ON response'
                shift = self.wind_direction_encoder.upwind_activation(current_dir, wind_dir)
            else:
                s = 'Random'
                shift = 0
        elif self.odour_navigation_strategies[strategy] == 2:
            s = 'OFF response'
            if self.handedness:
                shift = np.min([-delta_odour * oh_k, 3])
            else:
                shift = np.max([delta_odour * oh_k, -3])
        elif self.odour_navigation_strategies[strategy] == 3:            
            # generate final desired heading including the odour homing and upwind following
            # shifting signal from upwind direction - ON response
            if delta_odour > self.on_response_thr:
                s = 'ON response'
                shift = self.wind_direction_encoder.upwind_activation(current_dir, wind_dir)
            # shifting signal from odour homing - OFF response
            elif delta_odour > self.off_response_thr:
                if self.odour_on:
                    s = 'ON response'
                    shift = self.wind_direction_encoder.upwind_activation(current_dir, wind_dir)
                else:
                    s = 'Random'
                    shift = 0
            else:
                s = 'OFF response'
                if self.handedness:
                    shift = np.min([-delta_odour * oh_k, 3])
                else:
                    shift = np.max([delta_odour * oh_k, -3])
        else:
            s = 'NaN'
            shift = 0
            
        # shift the current heading
        o_w_dir = np.tile(neural_shift(self.cx.I_tb1, shift, self.wind_neuron_pref), 2)
        
#         return noisy_sigmoid(o_w_dir, 6.8, 3.0, self.noise), s 
        return o_w_dir, s
    
    def integration_output(self, with_pi=False, odour_strategy='None'):
        if with_pi:
            if odour_strategy == 'None':
                integration = self.cpu4_memory
            else:
                pi = self.cpu4_memory
                integration = self.ra.cue_integration_output(pi,
                                                             self.odour_gauidance_activation*self.odour_novelty*self.odour_tune_k)
        else:
            if odour_strategy == 'None':
                integration = self.cx.I_tb1
            else:
                integration = self.odour_gauidance_activation
        return integration

    def run(self, start_pos, start_h, time_out, w_dir, motor_k=1.0, step_size=0.04, odour_strategy='odour_gated_upwind', with_pi=False, constant_odour=False, odour_region=None):
        pos = np.zeros([time_out, 2])
        velocity = np.zeros([time_out, 2])
        h = np.zeros(time_out)
        pos[0] = start_pos
        h[0] = start_h

        odour_con = []
        delta_odour = np.zeros(time_out)
        pi_memory = np.zeros([time_out, 16])
        ra_memory = np.zeros([time_out, 16])
        
        states = []

        print("$-Start homing...")
        for t in range(time_out - 1):
            # update celestial current heading - TB1 neurons and path integration - PI
            self._update_pi_neuron_activation(h[t], velocity[t])
            pi_memory[t, :] = self.cpu4_memory
            
            # wind direction
            self.odour_field.wind_direction = w_dir[t]
            
            # update odour input and wind direction
            odour = self.odour_field.get_odour_concentration(pos[t, 0], pos[t, 1])
            if self.odour_field_add is not None:
                odour += self.odour_field_add.get_odour_concentration(pos[t, 0], pos[t, 1])
            self.odour_novelty = odour
            odour_con.append(self.odour_novelty)
            self.odour_on = odour_con[t] > self.odour_on_thr
            
             # delta odour
            delta_odour[t] = odour_con[t] - odour_con[t-1] if t > 0 else 0

            # update upwind direction activation
            self.update_upwind_activation(h[t], w_dir[t])
            
            # get odour navigation guidance activation 
            self.odour_gauidance_activation, state = self.generate_odour_wind_desired_heading(h[t], w_dir[t], delta_odour[t], odour_strategy, oh_k=10.0)
            
            states.append(state)
            if (state == 'Random') and (not with_pi):
                self.random_motor_w = 1.0
            else:
                self.random_motor_w = 0.1
            
            # heading from the random wandering (kind of search?)
            h_random = (np.random.rand(1) - 0.5) * (np.pi/2) * self.random_motor_w
            h_navigation = (self.cx.motor_value * motor_k)*(1 - self.random_motor_w)
                         
            # integrate PI and OW as final desired heading
            self.cx.desired_heading_memory = self.integration_output(with_pi, odour_strategy)
            
            # store integrated memory
            ra_memory[t, :] = self.cx.desired_heading_memory 

            # celestial heading as current heading
            self.cx.current_heading_memory = self.cx.I_tb1

            # steering circuit
            self.cx.steering_circuit_out()

            # moving forward
            h[t + 1] = (h[t] + np.pi + h_navigation + h_random) % (2.0 * np.pi) - np.pi
            velocity[t + 1, :] = np.array([np.cos(h[t + 1]), np.sin(h[t + 1])]) * step_size
            pos[t + 1, :] = pos[t, :] + velocity[t + 1, :]

            dis = np.sqrt(np.sum(pos[t + 1, 0] ** 2 + pos[t + 1, 1] ** 2))
            if dis < 0.15:
                print("$-End homing with nest distance %4.4s m" % dis)
                break

        return t, odour_con, pos, h, velocity, pi_memory, ra_memory, states, dis

class ShiftReferences(object):
    def __init__(self):
        self.cx = CentralComplexModel()
        self.random_motor_w = 0.1
        self.shift = 0
        
    def update_desired_headings(self, current_h=0, shift=0):
        # celestial heading as current heading
        self.cx.global_current_heading(current_h)
        self.cx.desired_heading_memory = np.tile(neural_shift(self.cx.I_tb1, 
                                                              shift, self.cx.phase_prefs), 2)
    
    def run(self, start_pos, start_h, time_out, motor_k=2.0, step_size=0.01):
        pos = np.zeros([time_out, 2])
        velocity = np.zeros([time_out, 2])
        h = np.zeros(time_out)
        pos[0] = start_pos
        h[0] = start_h
        for t in range(time_out - 1):
            self.cx.global_current_heading(h[t])
            # celestial heading as current heading
            self.cx.current_heading_memory = self.cx.I_tb1
            # steering circuit
            self.cx.steering_circuit_out()
            # heading from the random wandering (kind of search?)
            h_random = (np.random.rand(1) - 0.5) * (np.pi/2) * self.random_motor_w
            # combine the navigation and random noise
            h_navigation = (self.cx.motor_value * motor_k)*(1 - self.random_motor_w)
            # moving forward
            h[t + 1] = (h[t] + np.pi + h_navigation + h_random) % (2.0 * np.pi) - np.pi
            velocity[t + 1, :] = np.array([np.cos(h[t + 1]), np.sin(h[t + 1])]) * step_size
            pos[t + 1, :] = pos[t, :] + velocity[t + 1, :]
        
        return t, pos, h, velocity
        
def get_flow(heading, velocity, tn_prefs=np.pi / 4.0, filter_steps=0):
    """Calculate optic flow depending on preference angles. [L, R]"""

    A = np.array([[np.cos(heading + tn_prefs),
                   np.sin(heading + tn_prefs)],
                  [np.cos(heading - tn_prefs),
                   np.sin(heading - tn_prefs)]])
    flow = np.dot(A, velocity)

    return flow


def neural_shift(activation, shifting, pref):
    """
    shift the neural activation encoded direction
    :param activation:
    :param shifting:
    :param pref:
    :return:
    """
    roll = int(np.around(shifting, decimals=1) * 10)
    roll_high = int(roll / 10)
    roll_low = roll - roll_high*10
    temp = np.roll(activation, roll_high)
    if roll_low:
        # preference interpolate
        x1 = np.linspace(0, len(activation), len(activation), endpoint=False)
        x2 = np.linspace(0, len(activation), len(activation)*10, endpoint=False)
        prefs_tck = interpolate.splrep(x1, pref)
        
        new_preference = interpolate.splev(x2, prefs_tck)
        # activation interpolate
        act_tck = interpolate.splrep(pref, temp)
        new_activation = interpolate.splev(new_preference, act_tck)
        shifted = np.roll(new_activation, roll_low)[::10]
    else:
        shifted = temp
    return np.clip(shifted, 0, 1)


def decode_dir_from_activation(activation, prefs):
    return np.arctan2(np.sum(activation*np.sin(prefs)), np.sum(activation*np.cos(prefs)))
