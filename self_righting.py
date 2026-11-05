"""Motor-only sagittal self-righting; no simulation pose/velocity resets."""
import mujoco
import numpy as np

from closed_kinematics import wrap


class SelfRightingController:
    """Plant wheels, extend legs, catch with LQR, then verify a supported stand.

    The CAD linkage is not fore/aft symmetric. Rear falls need additional
    pitch damping during planting. Flat side-lying and fully inverted poses
    are outside this controller's validated operating region.
    """

    def __init__(self, jump, enabled=True):
        self.jump = jump
        self.c = jump.c
        self.enabled = enabled
        self.lengths = np.linspace(.10, .18, 41)
        self.poses = np.array([self.c.leg.ik(L, 0.) for L in self.lengths])
        self.reset()

    def reset(self):
        self.phase = 'IDLE'
        self.active = self.succeeded = self.failed = False
        self.reason = None
        self.stable_since = None
        self.started = self.phase_started = 0.
        self.attempts = 0

    def orientation(self, data):
        R = data.xmat[self.c.chassis].reshape(3, 3)
        pitch = float(np.arctan2(self.heading @ R[:2, 2], R[2, 2]))
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.c.model, data, mujoco.mjtObj.mjOBJ_BODY,
                                self.c.chassis, velocity, 1)
        return R, pitch, float(velocity[1])

    def fail(self, reason):
        self.reason = reason
        self.phase = 'FAILED'
        self.active = self.succeeded = False
        self.failed = True
        self.jump.failure = 'Self-righting: '+reason
        print(self.jump.failure+'; press G to retry or R to reset.', flush=True)

    def start(self, data):
        if not self.enabled:
            return False
        self.reset()
        R = data.xmat[self.c.chassis].reshape(3, 3)
        if self.jump.failure is None and R[2, 2] > np.cos(.08):
            info = self.jump.contact_info(data)
            if min(info['forces']) > 3. and not info['nonwheel']:
                print('Already standing; motor self-righting not needed.', flush=True)
                return False
        self.heading = np.array([R[1, 1], -R[0, 1]])
        self.heading /= max(np.linalg.norm(self.heading), 1e-9)
        _, pitch, _ = self.orientation(data)
        if abs(R[2, 1]) > .75:
            self.fail('side-lying pose outside validated range')
            return False
        if abs(pitch) > np.deg2rad(150):
            self.fail('fully inverted pose outside validated range')
            return False
        self.initial_pitch = pitch
        self.started = self.phase_started = float(data.time)
        self.active = True
        self.phase = 'PLANT'
        self.attempts = 1
        design = self.jump.scheduled(.18)
        self.c.length_ref = .18
        self.c.k_length, self.c.d_length = 1500., 45.
        self.c.support0 = float(design['support'])
        self.c.moment0, self.c.wheel0 = float(design['moment']), float(design['wheel'])
        self.c.yaw_ref = float(np.arctan2(self.heading[1], self.heading[0]))
        self.c.yaw_rate_ref = 0.
        self.jump.failure = 'Self-righting in progress'
        print('Motor self-righting: PLANT', flush=True)
        return True

    def update(self, data):
        try:
            return self._update(data)
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.fail(str(exc))
            return np.zeros(self.c.model.nu)

    def _update(self, data):
        zero = np.zeros(self.c.model.nu)
        if not self.active:
            return zero
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            self.fail('nonfinite state')
            return zero
        if data.time-self.started > 10.:
            self.fail('timed out without a stable wheel-supported stand')
            return zero
        R, pitch, rate = self.orientation(data)
        elapsed = float(data.time)-self.phase_started
        state = self.c.observe(data)
        floor = self.jump.surface_height(data.xpos[self.c.chassis, :2])
        if self.phase in ['PLANT', 'RISE']:
            self.phase = 'PLANT' if elapsed < .4 else 'RISE'
            length = .10+.08*float(np.clip((elapsed-.4)/.8, 0., 1.))
            base = np.array([np.interp(length, self.lengths, self.poses[:, i]) for i in range(2)])
            # At a shallow initial fall, use the wheel outside the chassis edge
            # if the planting motion rolls the body past horizontal.
            lean = np.sign(pitch)*1.2 if abs(self.initial_pitch) < np.pi/2 and abs(pitch) > 2.2 else 0.
            torque = zero.copy()
            for side in self.c.sides:
                q, qd = data.qpos[side['qp'][:2]], data.qvel[side['qv'][:2]]
                hip = -20.*wrap(base-pitch+lean-q)+.5*qd
                if self.initial_pitch < 0:
                    _, _, J = self.c.polar(q)
                    hip -= J[1]*(2.*pitch+.7*rate)
                torque[side['actuators'][:2]] = hip
                torque[side['actuators'][2]] = np.clip(2.*pitch+.4*rate, -2., 2.)
            if self.initial_pitch >= 0:
                ready = R[2, 2] > .92 and abs(state[1]) < .3 and data.qpos[2]-floor > .15
            else:
                ready = (R[2, 2] > .95 and abs(state[1]) < .5
                         and data.qpos[2]-floor > .14 and abs(rate) < 8.)
            if ready:
                self.phase = 'CATCH'
                self.catch_reference = float(state[0]-self.c.x0[0])
                self.phase_started = float(data.time)
                print('Motor self-righting: CATCH / brake and balance', flush=True)
            else:
                return np.clip(torque, self.c.model.actuator_ctrlrange[:, 0],
                               self.c.model.actuator_ctrlrange[:, 1])
        torque = self.c.inner(data, self.c.action(data, position_ref=self.catch_reference))
        info = self.jump.contact_info(data)
        lengths = self.c.last['lengths']
        standing = (R[2, 2] > np.cos(.08) and abs(state[3]) < .05
                    and max(abs(self.c.body_attitude(data)[3:])) < .3
                    and min(info['forces']) > 3. and not info['nonwheel']
                    and max(abs(np.asarray(lengths)-.18)) < .01)
        self.stable_since = (float(data.time) if self.stable_since is None else self.stable_since) if standing else None
        if self.stable_since is not None and data.time-self.stable_since >= .4:
            self.phase = 'COMPLETE'
            self.active = False
            self.succeeded = True
            print('Motor self-righting complete: both wheels support a stable stand.', flush=True)
        elif elapsed > 1.5 and info['nonwheel'] and data.qpos[2]-floor < .13 and self.attempts < 3:
            self.phase = 'PLANT'
            self.phase_started = float(data.time)
            self.attempts += 1
            print('Motor self-righting: retry planting from ground contact', flush=True)
        return torque
