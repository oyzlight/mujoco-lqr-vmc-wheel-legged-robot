"""Motor-driven trajectory for the known 15 cm platform; separate from flat hops."""
import numpy as np
from md.jump_controller import JumpController, smooth, smooth_rate, default_takeoff_momentum_gain


class StepJumpController(JumpController):
    def __init__(self, interactive=False, **kwargs):
        super().__init__(auto_jump=False, crouch=.08, extension=.23, thrust_time=.13,
                         boost=250., force_limit=400., cut_length=.205,
                         tuck_length=.08, tuck_duration=.055, deep_schedule=True, **kwargs)
        self.takeoff_momentum_target = -.005
        self.max_jump_speed = 1.05
        self.edge, self.top = 1.5, .15
        self.foot_offset = 0.
        self.landing_initial_length = .10
        self.landing_target = .18
        self.step_assist_available = interactive
        self.step_active = True
        if interactive:
            self.set_step_mode(False)

    def set_step_mode(self, enabled):
        self.step_active = enabled
        self.crouch_length = .08 if enabled else .12
        self.extension = .23
        self.thrust_time = .13 if enabled else .09
        self.boost = 250. if enabled else 50.
        self.cut_length = .205 if enabled else .215
        self.tuck_length = .08 if enabled else .11
        self.c.radial_force_limit = 400. if enabled else 80.
        self.takeoff_momentum_target = -.005 if enabled else 0.
        self.takeoff_momentum_gain = default_takeoff_momentum_gain(self.thrust_time)
        self.max_jump_speed = 1.05 if enabled else .6

    def update(self, data):
        if not self.step_active:
            return super().update(data)
        info = self.contact_info(data)
        t, elapsed = float(data.time), float(data.time)-self.entered
        values = [self.c.polar(data.qpos[s['qp'][:2]]) for s in self.c.sides]
        measured = float(np.mean([v[0] for v in values]))
        if self.phase == 'CROUCH' and elapsed >= .9 and abs(measured-self.crouch_length) < .004:
            self.transition('THRUST', data, info)
        elif self.phase == 'THRUST':
            clear = max(info['contact_forces']) < .5 and min(info['clearances']) > .002
            self.clear_since = (t if self.clear_since is None else self.clear_since) if clear else None
            if self.clear_since is not None and t-self.clear_since >= .005:
                self.transition('FLIGHT', data, info)
            elif elapsed > self.thrust_time+.3:
                self.failure = 'No confirmed step takeoff'
        elif self.phase == 'FLIGHT':
            # Keep wheels tucked while crossing the riser. Predict a short time
            # ahead to begin a controlled touchdown rather than a standing-length extension.
            x = min(info['wheel_positions'][:, 0])
            vx = float(info['com_velocity'][0])
            if x+max(0., vx)*.025 >= self.edge+.015 and info['com_vz'] < .05:
                self.transition('LANDING_PREP', data, info)
        if self.phase in ['FLIGHT', 'LANDING_PREP'] and self.flight_start is not None:
            if t-self.flight_start > .025 and min(info['forces']) > 3.:
                self.transition('LANDING', data, info)
                self.landing_initial_length = measured
        elif self.phase == 'LANDING' and elapsed > .7:
            self.transition('RECOVER', data, info)
        if info['nonwheel']:
            self.failure = 'Nonwheel terrain contact during step jump'
        elapsed = float(data.time)-self.entered
        if self.phase in ['BALANCE', 'RECOVER']:
            torque = self.ground_control(data, self.stand_length)
            self.set_air_action('NONE', data)
        elif self.phase == 'CROUCH':
            torque = self.ground_control(data, self.stand_length+(self.crouch_length-self.stand_length)*smooth(elapsed/.7))
        elif self.phase == 'THRUST':
            self.thrust_cut |= elapsed >= self.thrust_time or measured >= self.cut_length
            if not self.thrust_cut:
                speed = (self.extension-self.crouch_length)/self.thrust_time
                torque = self.ground_control(data, self.crouch_length+speed*elapsed,
                                            velocity_ref=speed, boost=self.boost, kd=25.)
            else:
                self.set_air_action('RELEASE', data)
                torque = self.air_control(data, .17)
        elif self.phase == 'FLIGHT':
            self.set_air_action('TUCK', data)
            progress = elapsed/self.tuck_duration
            length = self.air_start_length+(self.tuck_length-self.air_start_length)*smooth(progress)
            rate = (self.tuck_length-self.air_start_length)*smooth_rate(progress)/self.tuck_duration
            torque = self.air_control(data, length, float(np.clip(rate, -1.5, 1.5)), beta_offset=self.foot_offset)
        elif self.phase == 'LANDING_PREP':
            self.set_air_action('EXTEND', data)
            pitch = self.c.body_attitude(data)[1]
            target = min((data.xpos[s['thigh'], 2]-self.top-self.c.radius+.003)
                         /max(.65, np.cos(v[1]+pitch)) for s, v in zip(self.c.sides, values))
            target = float(np.clip(target, self.tuck_length, self.landing_target))
            torque = self.air_control(data, target, beta_offset=self.foot_offset)
        else:
            self.set_air_action('NONE', data)
            length = self.landing_initial_length+(self.stand_length-self.landing_initial_length)*smooth(elapsed/.5)
            torque = self.ground_control(data, length, kp=1100., kd=65.)
        self.last.update(phase=self.phase, air_action=self.air_action, **info)
        return torque


class StepAssist:
    """Known-scene run-up, jump and braking, requested by J from the spawn area."""

    def __init__(self, keys):
        self.keys = keys
        self.jump = keys.jump
        self.reset()

    def reset(self):
        self.active = False
        self.phase = 'IDLE'
        self.jump.set_step_mode(False)

    def begin(self, data):
        j = self.jump
        pos = np.mean([data.xpos[s['wheel'], :2] for s in j.c.sides], axis=0)
        state, att = j.c.observe(data), j.c.body_attitude(data)
        step = j.model.geom('step_1').id
        edge = j.model.geom_pos[step, 0]-j.model.geom_size[step, 0]
        height = j.model.geom_pos[step, 2]+j.model.geom_size[step, 2]
        if (j.failure or j.phase not in ['BALANCE', 'RECOVER']
                or abs(pos[0]-j.c.x0[0]) > .025 or abs(pos[1]) > .025
                or abs(state[3]) > .05 or max(abs(att[:3])) > .05
                or not np.isclose(edge, 1.5) or not np.isclose(height, .15)):
            print('Step assist: start upright and stopped near spawn, facing +X. Use R, then J.', flush=True)
            return False
        self.jump.set_step_mode(True)
        self.active, self.phase, self.started = True, 'APPROACH', float(data.time)
        self.keys.target_speed = .6
        self.keys.target_yaw_rate = 0.
        self.keys.pending_jump = False
        print('15 cm step assist: approach at 0.6 m/s; automatic jump and braking.', flush=True)
        return True

    def cancel(self):
        self.active = False
        self.phase = 'CANCELLED'
        # An already launched jump keeps its landing trajectory.
        if self.jump.phase in ['BALANCE', 'RECOVER']:
            self.jump.set_step_mode(False)
        self.keys.target_speed = self.keys.target_yaw_rate = 0.
        self.keys.pending_jump = False

    def update(self, data):
        if not self.active:
            if self.phase == 'CANCELLED' and self.jump.phase == 'RECOVER':
                self.jump.set_step_mode(False)
            return
        j = self.jump
        if j.failure:
            self.active, self.phase = False, 'FAILED'
            j.set_step_mode(False)
            self.keys.target_speed = 0.
            return
        x = float(np.mean([data.xpos[s['wheel'], 0] for s in j.c.sides]))
        if self.phase == 'APPROACH' and x >= .73:
            self.keys.pending_jump = True
            self.phase = 'EXECUTE'
        if (self.phase == 'EXECUTE' and j.phase in ['BALANCE', 'RECOVER']
                and self.keys.pending_jump and x > .80):
            self.cancel()
            print('Step assist cancelled: missed the takeoff preparation window.', flush=True)
            return
        if j.phase == 'LANDING':
            self.keys.target_speed = 0.
        if j.phase == 'RECOVER' and data.time-j.entered >= 3.:
            pos = np.array([data.xpos[s['wheel']] for s in j.c.sides])
            info = j.contact_info(data)
            good = (min(pos[:, 0]) > j.edge+.05 and max(pos[:, 0]) < 3.2
                    and max(abs(pos[:, 2]-.19)) < .005 and min(info['forces']) > 3.
                    and not info['nonwheel'] and max(abs(j.c.body_attitude(data)[:2])) < .05
                    and abs(j.c.observe(data)[3]) < .05 and data.qpos[2] > .34)
            self.active, self.phase = False, 'DONE' if good else 'FAILED'
            j.set_step_mode(False)
            if not good:
                j.failure = 'Step assist did not finish with a stable stand on the top'
            print('15 cm step assist: '+self.phase, flush=True)
        elif data.time-self.started > 11.:
            self.active, self.phase = False, 'FAILED'
            j.set_step_mode(False)
            j.failure = 'Step assist timed out'
