"""LQR + VMC for the corrected two-loop, FOUR HIP + TWO WHEEL robot.

Only physical motor torques are applied; no root forces or hidden position servos.
"""
from pathlib import Path
import json

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from closed_kinematics import ClosedLeg, wrap

ROOT = Path(__file__).resolve().parent


def rot_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s], [-s, c]])


def angle_difference(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))


class LqrVmc:
    """State: [axle_x, leg_lean, body_pitch, axle_vx, leg_rate, pitch_rate].

    LQR inputs: [total wheel torque, total virtual leg-angle moment].
    VMC supplies radial support and maps leg moments into A/B hip INPUT motor torques.
    Additional differential support / wheel torques stabilize roll / yaw.
    """

    def __init__(self, model_path=None, length=.18, control_dt=.005, wheel_contact_time=None):
        self.model = mujoco.MjModel.from_xml_path(str(model_path or ROOT / 'robot.xml'))
        if wheel_contact_time is not None:
            if wheel_contact_time < 2*self.model.opt.timestep:
                raise ValueError('Contact time must be at least twice the physics timestep')
            for side in ['left','right']:
                g=self.model.geom(side+'_wheel_collision').id
                self.model.geom_solref[g]=[wheel_contact_time,1.]
                # Equal-priority contacts average with the floor's 20 ms default.
                # Make this explicit override govern the wheel/floor pair.
                ground=self.model.geom('ground').id
                self.model.geom_priority[g]=max(self.model.geom_priority[g],self.model.geom_priority[ground]+1)
        self.length_ref = length
        self.leg = ClosedLeg()
        self.ik_seed = self.leg.ik(length, 0.)
        self.eq_offsets = np.zeros(9)
        self.nsub = int(round(control_dt / self.model.opt.timestep))
        if self.nsub < 1 or abs(self.nsub * self.model.opt.timestep - control_dt) > 1e-10:
            raise ValueError('Control period must be an integer multiple of physics timestep')
        self.dt = control_dt
        self.chassis = self.bid('chassis')
        self.sides = []
        for side in ['left', 'right']:
            names = ['hip_A', 'hip_B', 'output_A', 'output_B', 'rocker_pin',
                     'rod29_pin', 'lower_pin', 'rod35_pin', 'wheel_joint']
            joints = np.array([self.jid(side+'_'+n) for n in names])
            qp, qv = self.model.jnt_qposadr[joints], self.model.jnt_dofadr[joints]
            self.sides.append(dict(
                name=side, thigh=self.bid(side+'_Main_B'), shank=self.bid(side+'_Lower_Leg'),
                wheel=self.bid(side+'_Wheel'), joints=joints, all_qp=qp, all_qv=qv,
                qp=qp[[2,3,8]], qv=qv[[2,3,8]],
                actuators=np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                    side+'_'+n+'_motor') for n in ['hip_A','hip_B','wheel_joint']])))
        if self.model.jnt_type[self.jid('floating_base')] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError('This controller requires the free floating base in robot.xml')
        self.hip_local = self.model.body_pos[self.sides[0]['thigh']].copy()
        tyre = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'left_wheel_collision')
        self.radius = float(self.model.geom_size[tyre, 0])
        self.mass = float(self.model.body_mass.sum())
        self.k_length, self.d_length = 1500., 45.
        self.radial_force_limit = 80.
        self.k_roll, self.d_roll = 90., 5.
        self.k_yaw, self.d_yaw = .35, .08
        self.yaw_ref, self.yaw_rate_ref = 0., 0.
        self.k_sync, self.d_sync = 8., .4
        self.support0 = self.mass * 9.81 / 2
        self.moment0 = 0.
        self.wheel0 = 0.
        self.K = None
        self.q0 = None
        self.x0 = np.zeros(6)
        self.last = {}

    def bid(self, name):
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if i < 0:
            raise ValueError('Body missing from closed-loop model: '+name)
        return i

    def jid(self, name):
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if i < 0:
            raise ValueError('Joint missing: '+name+'; LQR runner requires the 3D robot.xml')
        return i

    def polar(self, q):
        """q is the TWO coaxial OUTPUT angles; J is the exact closed-loop Jacobian."""
        return self.leg.polar(q)

    def ik(self, length, beta):
        return self.leg.ik(length, beta, self.ik_seed)

    def leg_coordinates(self, q):
        return np.r_[-q[0], -q[1], self.leg.forward(q)['angles'], 0.]

    def set_leg_pose(self, data, side, q):
        data.qpos[side['all_qp']] = self.leg_coordinates(q)+self.eq_offsets

    def set_leg_velocity(self, data, side, q, qd):
        h = 1e-6
        t = np.column_stack([wrap(self.leg_coordinates(q+np.eye(2)[i]*h)-
                                  self.leg_coordinates(q-np.eye(2)[i]*h))/(2*h) for i in range(2)])
        data.qvel[side['all_qv']] = t@qd

    def body_attitude(self, data):
        R = data.xmat[self.chassis].reshape(3, 3)
        roll, pitch, yaw = Rotation.from_matrix(R).as_euler('xyz')
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY,
                                self.chassis, velocity, 1)
        wx, wy, wz = velocity[:3]
        pitch_rate = np.cos(roll) * wy - np.sin(roll) * wz
        yaw_rate = (np.sin(roll) * wy + np.cos(roll) * wz) / np.cos(pitch)
        roll_rate = wx + np.sin(roll) * np.tan(pitch) * wy + np.cos(roll) * np.tan(pitch) * wz
        return np.array([roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate])

    def body_jac(self, data, body):
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, data, jac, None, body)
        return jac

    def constraint_errors(self, data):
        closure = 0.
        gear = 0.
        for side in self.sides:
            name = side['name']
            for pin in ['C','D']:
                ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name+'_'+pin+str(i)) for i in [0,1]]
                closure = max(closure, float(np.linalg.norm(data.site_xpos[ids[0]]-data.site_xpos[ids[1]])))
            gear = max(gear, float(np.max(np.abs(data.qpos[side['all_qp'][:2]]+data.qpos[side['qp'][:2]]))))
        return closure, gear

    def observe(self, data):
        attitude = self.body_attitude(data)
        yaw, yaw_rate = attitude[2], attitude[5]
        heading = np.array([np.cos(yaw), np.sin(yaw)])
        heading_rate = yaw_rate * np.array([-np.sin(yaw), np.cos(yaw)])
        xs, vs, angles, rates = [], [], [], []
        for side in self.sides:
            hip = data.xpos[side['thigh']]
            wheel = data.xpos[side['wheel']]
            jhip, jwheel = self.body_jac(data, side['thigh']), self.body_jac(data, side['wheel'])
            delta = hip - wheel
            vdelta = (jhip - jwheel) @ data.qvel
            forward_delta = heading @ delta[:2]
            forward_rate = heading @ vdelta[:2] + heading_rate @ delta[:2]
            angles.append(np.arctan2(forward_delta, delta[2]))
            rates.append((delta[2] * forward_rate - forward_delta * vdelta[2]) /
                         (forward_delta**2 + delta[2]**2))
            xs.append(heading @ wheel[:2])
            vs.append(heading @ (jwheel @ data.qvel)[:2])
        return np.array([np.mean(xs), np.mean(angles), attitude[1],
                         np.mean(vs), np.mean(rates), attitude[4]])

    def inner(self, data, virtual, clip=True):
        """Virtual work: tau_output=J_closed.T@[F,M]; tau_input=-tau_output."""
        roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate = self.body_attitude(data)
        differential_support = -self.k_roll * roll - self.d_roll * roll_rate
        differential_wheel = (-self.k_yaw * angle_difference(yaw, self.yaw_ref)
                              - self.d_yaw * (yaw_rate-self.yaw_rate_ref))
        torque = np.zeros(self.model.nu)
        lengths, forces, conditions = [], [], []
        leg_values = []
        for side in self.sides:
            length, beta, J = self.polar(data.qpos[side['qp'][:2]])
            rates = J @ data.qvel[side['qv'][:2]]
            leg_values.append((length, beta, J, rates))
        beta_difference = angle_difference(leg_values[0][1], leg_values[1][1])
        rate_difference = leg_values[0][3][1] - leg_values[1][3][1]
        sync_moment = -self.k_sync * beta_difference - self.d_sync * rate_difference
        for i, side in enumerate(self.sides):
            length, beta, J, rates = leg_values[i]
            length_rate = rates[0]
            force = (self.support0 + self.k_length * (self.length_ref - length)
                     - self.d_length * length_rate + (1 if i == 0 else -1) * differential_support)
            if clip:
                # Virtual axial actuator force can pull as well as push;
                # it is NOT a unilateral ground normal force.
                force = float(np.clip(force, -self.radial_force_limit, self.radial_force_limit))
            moment = self.moment0 + virtual[1] / 2 + (1 if i == 0 else -1) * sync_moment
            torque[side['actuators'][:2]] = -J.T @ np.array([force, moment])
            wheel_torque = (self.wheel0 + virtual[0]) / 2
            wheel_torque += (-1 if i == 0 else 1) * differential_wheel / 2
            torque[side['actuators'][2]] = wheel_torque
            lengths.append(length)
            forces.append(force)
            conditions.append(np.linalg.cond(J))
        raw = torque.copy()
        if clip:
            torque = np.clip(torque, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        self.last = dict(lengths=lengths, forces=forces, jacobian_conditions=conditions,
                         attitude=attitude_list(roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate),
                         raw_motor_torque=raw, motor_torque=torque,
                         saturation=bool(np.any(abs(torque - raw) > 1e-9)))
        return torque

    def solve_equilibrium(self):
        """Body pitch=0, desired leg length; solve actual 3D contact statics."""
        data = mujoco.MjData(self.model)

        def put(p):
            height, theta, support, moment, wheel = p[:5]
            # Soft equality constraints need small static deflections to carry load.
            self.eq_offsets[:] = 0
            self.eq_offsets[[0,1,4,5,6,7]] = p[5:]
            mujoco.mj_resetData(self.model, data)
            data.qpos[:3] = [0, 0, height]
            data.qpos[3:7] = [1, 0, 0, 0]
            for side in self.sides:
                self.set_leg_pose(data, side, self.ik(self.length_ref, theta))
            self.support0, self.moment0, self.wheel0 = support, moment, wheel
            mujoco.mj_forward(self.model, data)
            data.ctrl[:] = self.inner(data, np.zeros(2), clip=False)
            mujoco.mj_forward(self.model, data)
            return data.qacc.copy()

        # Near-contact start: wheel axle is L below its hip at theta=0.
        height = self.radius + self.length_ref - self.hip_local[2] - .0001
        fit = least_squares(put, [height, 0., self.mass*9.81/2, 0., 0., *np.zeros(6)],
                            bounds=([height-.02, -.4, 0, -4, -.5, *np.full(6,-.01)],
                                    [height+.02, .4, 50, 4, .5, *np.full(6,.01)]),
                            xtol=1e-12, ftol=1e-12, gtol=1e-10, max_nfev=500,
                            x_scale=[.2, .1, 20, 1, .1, *np.full(6,.001)])
        residual = put(fit.x)
        if max(abs(residual)) > 1e-4:
            raise RuntimeError(f'Static equilibrium failed: {fit.x}, qacc={residual}')
        self.q0 = data.qpos.copy()
        self.geometric_beta0 = float(fit.x[1])
        self.x0 = self.observe(data)
        self.axle_z0 = np.mean([data.xpos[s['wheel'], 2] for s in self.sides])
        if np.any(abs(data.ctrl) > self.model.actuator_ctrlrange[:, 1]):
            raise RuntimeError('Equilibrium requires excessive motor torque')
        self.equilibrium_info = dict(parameters=fit.x.tolist(),
                                     parameter_order=['root_z','ideal_leg_lean','radial_feedforward','leg_moment_feedforward','wheel_feedforward',
                                                      'input_A_deflection','input_B_deflection','rocker_deflection','rod29_deflection','lower_deflection','rod35_deflection'],
                                     soft_constraint_offsets=self.eq_offsets.tolist(),
                                     max_acceleration=float(max(abs(residual))),
                                     state=self.x0.tolist(), qpos=self.q0.tolist(),
                                     motor_torque=data.ctrl.tolist())
        return data

    def initialize(self, data, dx=None):
        """Lift six independent sagittal states to a contact-consistent 3D pose."""
        x = self.x0 + (np.zeros(6) if dx is None else np.asarray(dx))
        position, theta, pitch, velocity, theta_rate, pitch_rate = x
        mujoco.mj_resetData(self.model, data)
        data.qpos[:] = self.q0
        quat = Rotation.from_euler('y', pitch).as_quat()
        data.qpos[3:7] = quat[[3, 0, 1, 2]]
        for side in self.sides:
            q = self.ik(self.length_ref, self.geometric_beta0 + theta-self.x0[1]-pitch)
            self.set_leg_pose(data, side, q)
            _, _, J = self.polar(q)
            self.set_leg_velocity(data, side, q, np.linalg.solve(J, [0, theta_rate-pitch_rate]))
        mujoco.mj_forward(self.model, data)
        axle = np.mean([data.xpos[s['wheel']] for s in self.sides], axis=0)
        data.qpos[0] += position - axle[0]
        data.qpos[2] += self.axle_z0 - axle[2]
        data.qvel[4] = pitch_rate
        mujoco.mj_forward(self.model, data)
        jac = np.mean([self.body_jac(data, s['wheel']) for s in self.sides], axis=0)
        axle_velocity = jac @ data.qvel
        data.qvel[0] += velocity - axle_velocity[0]
        data.qvel[2] -= axle_velocity[2]
        for side in self.sides:
            qd = data.qvel[side['qv'][:2]]
            data.qvel[side['qv'][2]] = velocity/self.radius - pitch_rate - data.qvel[side['all_qv'][3]] - data.qvel[side['all_qv'][6]]
        mujoco.mj_forward(self.model, data)

    def linearize(self, eps=1e-5):
        scratch = mujoco.MjData(self.model)

        def step(dx, du):
            self.initialize(scratch, dx)
            # Outer virtual action held; radial VMC and roll/yaw feedback run each physics step.
            for _ in range(self.nsub):
                scratch.ctrl[:] = self.inner(scratch, du)
                mujoco.mj_step(self.model, scratch)
                mujoco.mj_forward(self.model, scratch)
            return self.observe(scratch) - self.x0

        A = np.column_stack([(step(np.eye(6)[i]*eps, np.zeros(2))
                              - step(-np.eye(6)[i]*eps, np.zeros(2))) / (2*eps)
                             for i in range(6)])
        B = np.column_stack([(step(np.zeros(6), np.eye(2)[i]*eps)
                              - step(np.zeros(6), -np.eye(2)[i]*eps)) / (2*eps)
                             for i in range(2)])
        rank = np.linalg.matrix_rank(np.hstack([np.linalg.matrix_power(A, i)@B for i in range(6)]))
        if rank != 6:
            raise RuntimeError(f'Reduced controllability rank {rank}, expected 6')
        Q = np.diag([30., 200., 300., 12., 10., 10.])
        R = np.diag([.4, .8])
        P = solve_discrete_are(A, B, Q, R)
        self.K = np.linalg.solve(R+B.T@P@B, B.T@P@A)
        poles = np.linalg.eigvals(A-B@self.K)
        if max(abs(poles)) >= 1:
            raise RuntimeError('Discrete LQR closed-loop pole outside the unit circle')
        self.design_info = dict(A=A.tolist(), B=B.tolist(), Q=Q.tolist(), R=R.tolist(),
                                K=self.K.tolist(), controllability_rank=int(rank),
                                poles=[[float(v.real), float(v.imag)] for v in poles],
                                max_pole_magnitude=float(max(abs(poles))),
                                linearization_fixed_point_drift=step(np.zeros(6), np.zeros(2)).tolist(),
                                control_dt=self.dt, physics_dt=self.model.opt.timestep)
        return self.design_info

    def action(self, data, position_ref=0., velocity_ref=0., lqr=True):
        error = self.observe(data) - self.x0
        error[0] -= position_ref
        error[3] -= velocity_ref
        virtual = -self.K @ error if lqr else np.zeros(2)
        return virtual

    def save_design(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        info = dict(model=str(ROOT/'robot.xml'), state_order=['axle_x', 'leg_lean', 'body_pitch',
                    'axle_vx', 'leg_rate', 'pitch_rate'],
                    input_order=['total_wheel_torque', 'total_virtual_leg_moment'],
                    leg_length_m=self.length_ref, mass_kg=self.mass,
                    radial_gains=[self.k_length, self.d_length],
                    roll_gains=[self.k_roll, self.d_roll], yaw_gains=[self.k_yaw, self.d_yaw],
                    leg_synchronization_gains=[self.k_sync, self.d_sync],
                    equilibrium=self.equilibrium_info, lqr=self.design_info)
        info['topology']='closed-two-loop-four-hip-v1'
        info['motor_mapping']='tau_hip_inputs = -J_closed_output.T @ [radial_force, leg_moment]; 1:1 external gears'
        (directory/'design.json').write_text(json.dumps(info, indent=2), encoding='utf-8')


def attitude_list(*args):
    return [float(a) for a in args]
