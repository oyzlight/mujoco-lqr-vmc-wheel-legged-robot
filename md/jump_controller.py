"""Contact-driven single-hop state machine for the CAD two-loop robot.xml.

Only six motor torques are applied. No external lift, reset, or root constraints.
Grounded LQR gains are scheduled by measured leg length; they are disabled in air.
"""
import mujoco
import numpy as np

from lqr_vmc import LqrVmc

CONTROLLER_VERSION='closed-two-loop-four-hip-v1'


def smooth(value):
    t = np.clip(value, 0., 1.)
    return float(t*t*t*(10+t*(-15+6*t)))


def smooth_rate(value):
    t=float(np.clip(value,0.,1.))
    return 30*t*t*(1-t)*(1-t)


def default_takeoff_momentum_gain(thrust_time):
    if thrust_time<=0:
        raise ValueError('Thrust time must be positive')
    return 180.*float(np.clip(.10/thrust_time,1.,2.))


class JumpController:
    PHASES = ['BALANCE', 'CROUCH', 'THRUST', 'FLIGHT', 'LANDING_PREP', 'LANDING', 'RECOVER']
    AIR_ACTIONS=['NONE','RELEASE','TUCK','EXTEND']

    def __init__(self, crouch=.14, extension=.225, thrust_time=.10, boost=20., cut_length=None,
                 tuck_length=.13, force_limit=80., deep_schedule=False,wheel_contact_time=None,progress=None,
                 auto_jump=True, tuck_duration=.075, model_path=None):
        if tuck_duration <= 0:
            raise ValueError('Tuck duration must be positive')
        if progress:progress(.18)
        self.c = LqrVmc(model_path=model_path, wheel_contact_time=wheel_contact_time)
        self.c.radial_force_limit = float(force_limit)
        self.c.solve_equilibrium()
        self.c.linearize()
        self.model = self.c.model
        self.stand_length = .18
        self.crouch_length, self.extension = crouch, extension
        self.thrust_time, self.boost = thrust_time, boost
        self.cut_length=min(extension-.01,.215) if cut_length is None else cut_length
        self.schedule = []
        schedule_lengths=([.08,.10,.12,.14,.16,.18,.20,.22] if deep_schedule
                          else [.14,.16,.18,.20,.22])
        if crouch < min(schedule_lengths):
            schedule_lengths = sorted(set([crouch, *schedule_lengths]))
        for length in schedule_lengths:
            c = self.c if length == .18 else LqrVmc(model_path=model_path, length=length,wheel_contact_time=wheel_contact_time)
            if c is not self.c:
                if progress:progress(length)
                c.solve_equilibrium()
                c.linearize()
            self.schedule.append(dict(length=length, K=c.K.copy(), x0=c.x0.copy(),
                                      support=c.support0, moment=c.moment0, wheel=c.wheel0))
        self.reference_x = self.c.x0[0]
        self.reference_xy = None
        self.horizontal_velocity_ref = 0.
        self.auto_jump = auto_jump
        self.phase, self.entered = 'BALANCE', 0.
        self.events = [dict(time=0., phase=self.phase)]
        self.failure = None
        self.flight_start = self.landing_time = None
        self.flight_com_z = self.flight_vz = None
        self.clear_since = None
        self.thrust_cut = False
        self.virtual = np.zeros(2)
        self.takeoff_momentum_gain=default_takeoff_momentum_gain(thrust_time)
        self.flight_pitch_momentum=None
        self.tuck_length=tuck_length
        self.tuck_duration=tuck_duration
        # Re-tuned for the closed linkage: leave the ground near zero pitch momentum.
        self.takeoff_momentum_target=0.
        self.air_action='NONE'
        self.air_events=[]
        self.air_start_length=None
        self.landing_start_length=None
        self.landing_length=.19
        standing=mujoco.MjData(self.model)
        self.c.initialize(standing)
        self.touchdown_com_height=float(standing.subtree_com[self.c.chassis,2])+.008
        self.last_outer = -100.
        self.last = {}
        self.ground = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'ground')
        self.terrain = {g for g in range(self.model.ngeom)
                        if self.model.geom_bodyid[g] == 0 and self.model.geom_contype[g] != 0}
        self.wheels = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, side+'_wheel_collision')
                       for side in ['left','right']]

    def initialize(self, data):
        self.c.initialize(data)

    def request_jump(self, data, velocity_ref=None):
        """Start a supported hop; an explicit velocity target allows moving takeoff."""
        if self.failure or self.phase not in ['BALANCE', 'RECOVER']:
            return False
        if self.phase == 'RECOVER' and data.time-self.entered < .35:
            return False
        info = self.contact_info(data)
        state = self.c.observe(data)
        attitude = self.c.body_attitude(data)
        lengths = [self.c.polar(data.qpos[s['qp'][:2]])[0] for s in self.c.sides]
        velocity_ready = (abs(state[3]) <= .03 if velocity_ref is None else
                          abs(state[3]-velocity_ref) <= .15 and abs(state[3]) <= getattr(self, 'max_jump_speed', .6))
        if (min(info['forces']) < 3. or info['nonwheel']
                or not velocity_ready or abs(info['com_vz']) > .03
                or max(abs(attitude[:2])) > .08 or max(abs(attitude[3:5])) > .1
                or abs(attitude[5]) > .03
                or max(abs(np.asarray(lengths)-self.stand_length)) > .005):
            return False
        self.reference_x = float(state[0])
        if self.reference_xy is not None:
            self.reference_xy = np.mean([data.xpos[s['wheel'], :2] for s in self.c.sides], axis=0)
        self.horizontal_velocity_ref = 0. if velocity_ref is None else float(velocity_ref)
        self.flight_start = self.landing_time = None
        self.flight_com_z = self.flight_vz = self.flight_pitch_momentum = None
        self.clear_since = None
        self.thrust_cut = False
        self.air_start_length = self.landing_start_length = None
        self.transition('CROUCH', data, info)
        return True

    def surface_height(self, xy):
        """Top of static axis-aligned terrain boxes, or the original z=0 floor."""
        height = 0.
        for g in self.terrain:
            if self.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
                continue
            p, half = self.model.geom_pos[g], self.model.geom_size[g]
            if np.all(abs(np.asarray(xy)-p[:2]) <= half[:2]):
                height = max(height, float(p[2]+half[2]))
        return height

    def contact_info(self, data):
        forces = np.zeros(2)
        contact_forces = np.zeros(2)
        nonwheel = False
        wrench = np.zeros(6)
        for i, contact in enumerate(data.contact):
            if contact.geom1 not in self.terrain and contact.geom2 not in self.terrain:
                continue
            other = contact.geom2 if contact.geom1 in self.terrain else contact.geom1
            if other in self.wheels:
                mujoco.mj_contactForce(self.model, data, i, wrench)
                contact_forces[self.wheels.index(other)] += max(0., wrench[0])
                # A vertical riser hit does not count as upward wheel support.
                forces[self.wheels.index(other)] += max(0., wrench[0])*abs(contact.frame[2])
            else:
                nonwheel = True
        wheel_positions = np.array([data.xpos[s['wheel']] for s in self.c.sides])
        clearances = [p[2]-self.c.radius-self.surface_height(p[:2]) for p in wheel_positions]
        jac = np.zeros((3,self.model.nv))
        mujoco.mj_jacSubtreeCom(self.model,data,jac,self.c.chassis)
        com_velocity = jac @ data.qvel
        return dict(forces=forces, clearances=np.array(clearances),
                    contact_forces=contact_forces,
                    wheel_positions=wheel_positions, com_velocity=com_velocity,
                    com_z=float(data.subtree_com[self.c.chassis,2]),
                    com_vz=float(com_velocity[2]), nonwheel=nonwheel)

    def transition(self, phase, data, info):
        self.phase, self.entered = phase, float(data.time)
        self.last_outer = -100.
        self.events.append(dict(time=float(data.time), phase=phase,
                                com_z=info['com_z'], com_vz=info['com_vz']))
        if phase == 'FLIGHT':
            self.flight_start = float(data.time)
            self.flight_com_z, self.flight_vz = info['com_z'], info['com_vz']
            mujoco.mj_subtreeVel(self.model,data)
            yaw = self.c.body_attitude(data)[2]
            self.flight_pitch_momentum=float(data.subtree_angmom[self.c.chassis] @
                                             np.array([-np.sin(yaw), np.cos(yaw), 0.]))
            self.events[-1]['pitch_momentum_Nms']=self.flight_pitch_momentum
            self.air_start_length=float(np.mean([self.c.polar(data.qpos[s['qp'][:2]])[0]
                                                 for s in self.c.sides]))
        if phase=='LANDING_PREP':
            self.landing_start_length=float(np.mean([self.c.polar(data.qpos[s['qp'][:2]])[0]
                                                     for s in self.c.sides]))
        if phase == 'LANDING':
            self.landing_time = float(data.time)

    def scheduled(self, length):
        xs=np.array([p['length'] for p in self.schedule])
        index=int(np.clip(np.searchsorted(xs,length)-1,0,len(xs)-2))
        a,b=self.schedule[index:index+2]
        weight=float(np.clip((length-a['length'])/(b['length']-a['length']),0,1))
        return {k:(1-weight)*a[k]+weight*b[k] for k in ['K','x0','support','moment','wheel']}

    def ground_control(self, data, length_ref, velocity_ref=0., boost=0., kp=1500., kd=45.):
        measured=np.mean([self.c.polar(data.qpos[s['qp'][:2]])[0] for s in self.c.sides])
        design=self.scheduled(measured)
        if data.time-self.last_outer >= self.c.dt-1e-10:
            state = self.c.observe(data)
            error=state-design['x0']
            error[0]=state[0]-self.reference_x
            if self.reference_xy is not None:
                yaw = self.c.body_attitude(data)[2]
                error[0] = state[0] - np.array([np.cos(yaw), np.sin(yaw)]) @ self.reference_xy
            error[3]-=self.horizontal_velocity_ref
            self.virtual=-design['K']@error
            self.last_outer=float(data.time)
        # VMC velocity feed-forward has units N: kd * desired extension speed.
        self.c.length_ref=length_ref
        self.c.support0=float(design['support']+boost+kd*velocity_ref)
        self.c.moment0=float(design['moment'])
        self.c.wheel0=float(design['wheel'])
        self.c.k_length,self.c.d_length=kp,kd
        virtual=self.virtual.copy()
        if self.phase=='THRUST':
            # Fast centroidal feedback: do not leave the floor carrying a large
            # pitch angular momentum that has to be caught by rolling afterward.
            # This runs at the physics rate, independently of the slower LQR.
            mujoco.mj_subtreeVel(self.model,data)
            target=self.takeoff_momentum_target*smooth((data.time-self.entered)/(.5*self.thrust_time))
            yaw = self.c.body_attitude(data)[2]
            momentum = data.subtree_angmom[self.c.chassis] @ np.array([-np.sin(yaw), np.cos(yaw), 0.])
            virtual[0]+=self.takeoff_momentum_gain*(momentum-target)
        torque=self.c.inner(data,virtual)
        self.last=dict(self.c.last, length_ref=length_ref, virtual=virtual)
        return torque

    def air_control(self, data, length_ref, length_rate_ref=0., beta_offset=0.):
        # No grounded LQR, gravity-support feed-forward, or ground roll/yaw loop.
        roll,pitch,yaw,roll_rate,pitch_rate,yaw_rate=self.c.body_attitude(data)
        torque=np.zeros(self.model.nu)
        values=[]
        for side in self.c.sides:
            L,beta,J=self.c.polar(data.qpos[side['qp'][:2]])
            rates=J@data.qvel[side['qv'][:2]]
            values.append((L,beta,J,rates))
        beta_diff=values[0][1]-values[1][1]
        rate_diff=values[0][3][1]-values[1][3][1]
        sync=-4*beta_diff-.2*rate_diff
        for i,(side,(L,beta,J,rates)) in enumerate(zip(self.c.sides,values)):
            kp,kd=(450.,9.) if self.phase=='THRUST' or self.tuck_length is None else (650.,20.)
            F=np.clip(kp*(length_ref-L)+kd*(length_rate_ref-rates[0]),-60,60)
            beta_ref=self.c.x0[1]-pitch
            beta_ref += beta_offset
            if self.phase == 'LANDING_PREP':
                # Reach slightly along travel before touchdown, rather than
                # placing the wheel behind a step edge while the body advances.
                beta_ref -= float(np.clip(.3*self.horizontal_velocity_ref, -.15, .15))
            angle_kp,angle_kd=(.5,.04) if self.tuck_length is None else (1.5,.12)
            M=4.5*pitch+.5*pitch_rate-angle_kp*(beta-beta_ref)-angle_kd*rates[1]
            M+=(1 if i==0 else -1)*sync
            # External 1:1 gears reverse output angles AND output torque signs.
            torque[side['actuators'][:2]]=-J.T@np.array([F,M])
            # Reduce landing slip by matching absolute rolling speed, not by
            # accelerating the wheels with a ground-contact LQR while airborne.
            wheel_rate=data.qvel[side['qv'][2]]
            absolute_rate=(wheel_rate+data.qvel[side['all_qv'][3]]
                           +data.qvel[side['all_qv'][6]]+pitch_rate)
            velocity = self.c.body_jac(data,side['wheel'])@data.qvel
            vx = np.array([np.cos(yaw), np.sin(yaw)]) @ velocity[:2]
            torque[side['actuators'][2]]=-.008*(absolute_rate-vx/self.c.radius)
        raw=torque.copy()
        torque=np.clip(torque,self.model.actuator_ctrlrange[:,0],self.model.actuator_ctrlrange[:,1])
        self.last=dict(lengths=[v[0] for v in values],length_ref=length_ref,
                       motor_torque=torque,raw_motor_torque=raw,virtual=np.zeros(2),
                       saturation=bool(np.any(abs(raw-torque)>1e-9)))
        return torque

    def time_to_touchdown(self,info):
        remaining = .10
        for _ in range(3):
            height = max(self.surface_height(p[:2]+info['com_velocity'][:2]*remaining)
                         for p in info['wheel_positions'])
            dz=info['com_z']-self.touchdown_com_height-height
            discriminant=info['com_vz']**2+2*9.81*dz
            remaining=max(0.,(info['com_vz']+np.sqrt(max(0.,discriminant)))/9.81)
        return remaining

    def set_air_action(self,action,data):
        if action!=self.air_action:
            self.air_action=action
            self.air_events.append(dict(time=float(data.time),action=action))

    def update(self, data):
        info=self.contact_info(data)
        t=float(data.time); elapsed=t-self.entered
        measured=np.mean([self.c.polar(data.qpos[s['qp'][:2]])[0] for s in self.c.sides])
        if self.auto_jump and self.phase=='BALANCE' and t>=2.:
            self.transition('CROUCH',data,info)
        elif self.phase=='CROUCH' and elapsed>=.9 and abs(measured-self.crouch_length)<.004:
            self.transition('THRUST',data,info)
        elif self.phase=='THRUST':
            clear=bool(max(info['contact_forces'])<.5 and min(info['clearances'])>.002)
            if clear:
                self.clear_since=t if self.clear_since is None else self.clear_since
            else:
                self.clear_since=None
            if self.clear_since is not None and t-self.clear_since>=.005:
                self.transition('FLIGHT',data,info)
            elif elapsed>self.thrust_time+.3:
                self.failure='No confirmed two-wheel takeoff after thrust'
        elif (self.phase=='FLIGHT' and info['com_vz']<-.08
              and (self.tuck_length is None or
                   (elapsed>=.055 and self.time_to_touchdown(info)<.10))):
            self.transition('LANDING_PREP',data,info)
        elif self.phase in ['FLIGHT','LANDING_PREP']:
            if self.flight_start is not None and t-self.flight_start>.025 and max(info['forces'])>3.:
                self.transition('LANDING',data,info)
        elif self.phase=='LANDING' and elapsed>.6:
            self.transition('RECOVER',data,info)
        if info['nonwheel']:
            self.failure='Chassis or leg touched ground'
        elapsed=t-self.entered
        self.set_air_action('NONE' if self.phase not in ['THRUST','FLIGHT','LANDING_PREP']
                            else self.air_action,data)
        if self.phase=='BALANCE' or self.phase=='RECOVER':
            torque=self.ground_control(data,self.stand_length)
        elif self.phase=='CROUCH':
            length=self.stand_length+(self.crouch_length-self.stand_length)*smooth(elapsed/.7)
            torque=self.ground_control(data,length)
        elif self.phase=='THRUST':
            # Begin active retraction before the closed linkage reaches full
            # extension; a time-only thrust can overshoot toward a singularity.
            if elapsed>=self.thrust_time or measured>=self.cut_length:
                self.thrust_cut=True
            if not self.thrust_cut:
                speed=(self.extension-self.crouch_length)/self.thrust_time
                torque=self.ground_control(data,self.crouch_length+speed*elapsed,
                                           velocity_ref=speed,boost=self.boost,kd=25.)
            else:
                self.set_air_action('RELEASE',data)
                torque=self.air_control(data,.17)
        elif self.phase=='FLIGHT':
            self.set_air_action('TUCK' if self.tuck_length is not None else 'RELEASE',data)
            if self.tuck_length is None:
                torque=self.air_control(data,.17)
            else:
                duration=self.tuck_duration
                progress=elapsed/duration
                L=self.air_start_length+(self.tuck_length-self.air_start_length)*smooth(progress)
                Ldot=(self.tuck_length-self.air_start_length)*smooth_rate(progress)/duration
                torque=self.air_control(data,L,float(np.clip(Ldot,-1.5,1.5)))
        elif self.phase=='LANDING_PREP':
            self.set_air_action('EXTEND',data)
            if self.tuck_length is None:
                torque=self.air_control(data,.20)
            else:
                duration=.075
                progress=elapsed/duration
                L=self.landing_start_length+(self.landing_length-self.landing_start_length)*smooth(progress)
                Ldot=(self.landing_length-self.landing_start_length)*smooth_rate(progress)/duration
                torque=self.air_control(data,L,float(np.clip(Ldot,-1.5,1.5)))
        else:
            torque=self.ground_control(data,self.stand_length,kp=1100.,kd=65.)
        self.last.update(phase=self.phase,air_action=self.air_action,**info)
        return torque
