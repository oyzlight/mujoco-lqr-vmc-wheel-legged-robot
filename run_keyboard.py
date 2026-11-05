"""Keyboard driving, moving jumps and motor self-righting beside a 15 cm step."""
import argparse
from queue import Empty, SimpleQueue
import time

import mujoco
import numpy as np

from md.jump_controller import JumpController
from motor_limits import configure, limit
from lqr_vmc import ROOT
from self_righting import SelfRightingController
from step_jump import StepJumpController, StepAssist

# GLFW key codes used by MuJoCo's viewer callback.
KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262


class KeyboardControl:
    """Viewer thread queues keys; only the simulation thread changes commands."""

    def __init__(self, jump, max_speed=.4, self_righting=True):
        self.jump = jump
        self.max_speed = max_speed
        self.keys = SimpleQueue()
        self.target_speed = self.speed = 0.
        self.target_yaw_rate = self.yaw_rate = 0.
        self.pending_jump = False
        self.last_update = None
        # More responsive heading control for interactive driving.
        jump.c.k_yaw, jump.c.d_yaw = .7, .2
        self.righting = SelfRightingController(jump, enabled=self_righting)
        self.step_assist = StepAssist(self) if getattr(jump, 'step_assist_available', False) else None

    def reset(self, data):
        # Reset the existing controller without repeating equilibrium/LQR design.
        while True:
            try:
                self.keys.get_nowait()
            except Empty:
                break
        j = self.jump
        self.righting.reset()
        if self.step_assist:
            self.step_assist.reset()
        j.c.length_ref = j.stand_length
        j.initialize(data)
        j.phase, j.entered = 'BALANCE', 0.
        j.events = [dict(time=0., phase='BALANCE')]
        j.air_events = []
        j.air_action = 'NONE'
        j.failure = None
        j.flight_start = j.landing_time = None
        j.flight_com_z = j.flight_vz = j.flight_pitch_momentum = None
        j.clear_since = None
        j.thrust_cut = False
        j.air_start_length = j.landing_start_length = None
        j.last_outer = -100.
        j.virtual[:] = 0.
        j.reference_x = float(j.c.observe(data)[0])
        j.reference_xy = np.mean([data.xpos[s['wheel'], :2] for s in j.c.sides], axis=0)
        j.c.yaw_ref = j.c.yaw_rate_ref = 0.
        j.horizontal_velocity_ref = 0.
        self.target_speed = self.speed = 0.
        self.target_yaw_rate = self.yaw_rate = 0.
        self.pending_jump = False
        self.last_update = None

    def update(self, data):
        j = self.jump
        while True:
            try:
                key = self.keys.get_nowait()
            except Empty:
                break
            if key == ord('R'):
                self.reset(data)
                print('Reset to standing.', flush=True)
            elif key == ord('J') and self.step_assist and not self.righting.active:
                if not self.step_assist.active:
                    self.step_assist.begin(data)
            elif key == ord('G') and not self.righting.active:
                if self.step_assist:
                    self.step_assist.cancel()
                self.target_speed = self.speed = self.target_yaw_rate = self.yaw_rate = 0.
                self.pending_jump = False
                self.righting.start(data)
            elif self.righting.active:
                continue
            elif key == ord('X'):
                if self.step_assist and self.step_assist.active:
                    self.step_assist.cancel()
                self.target_speed = 0.
                self.target_yaw_rate = 0.
                self.pending_jump = False
            elif self.step_assist and self.step_assist.active:
                continue
            elif key == ord(' ') and not j.failure:
                if j.phase in ['BALANCE', 'RECOVER']:
                    self.target_yaw_rate = 0.
                    self.pending_jump = True
                    print('Jump requested: keeping travel speed, settling heading.', flush=True)
            elif key in (KEY_UP, KEY_DOWN) and not j.failure:
                if j.phase in ['BALANCE', 'RECOVER'] and not self.pending_jump:
                    self.target_speed = float(np.clip(
                        self.target_speed + (.15 if key == KEY_UP else -.15),
                        -self.max_speed, self.max_speed))
                    print(f'Target speed: {self.target_speed:+.2f} m/s', flush=True)
            elif key in (KEY_LEFT, KEY_RIGHT) and not j.failure:
                if j.phase in ['BALANCE', 'RECOVER'] and not self.pending_jump:
                    self.target_yaw_rate = float(np.clip(
                        self.target_yaw_rate + (.5 if key == KEY_LEFT else -.5), -1.2, 1.2))
                    print(f'Target turn rate: {self.target_yaw_rate:+.2f} rad/s', flush=True)
        if self.step_assist:
            self.step_assist.update(data)
        if j.failure or self.righting.active:
            return
        grounded = j.phase in ['BALANCE', 'RECOVER']
        dt = j.c.dt if self.last_update is None else max(0., float(data.time)-self.last_update)
        self.last_update = float(data.time)
        if grounded or j.phase == 'LANDING':
            self.speed += float(np.clip(self.target_speed-self.speed, -.6*dt, .6*dt))
            # Bound position lead so an obstructed robot cannot accumulate a large command.
            yaw = j.c.body_attitude(data)[2]
            self.yaw_rate += float(np.clip(self.target_yaw_rate-self.yaw_rate, -1.5*dt, 1.5*dt))
            j.c.yaw_rate_ref = self.yaw_rate
            yaw_error = np.arctan2(np.sin(j.c.yaw_ref+self.yaw_rate*dt-yaw),
                                   np.cos(j.c.yaw_ref+self.yaw_rate*dt-yaw))
            j.c.yaw_ref = yaw + float(np.clip(yaw_error, -.35, .35))
        # Carry the moving position/velocity target through crouch, flight and landing.
        # In flight this only advances the landing reference; there is no ground LQR.
        j.horizontal_velocity_ref = self.speed
        yaw = j.c.body_attitude(data)[2]
        heading = np.array([np.cos(yaw), np.sin(yaw)])
        position = np.mean([data.xpos[s['wheel'], :2] for s in j.c.sides], axis=0)
        j.reference_xy += self.speed*dt*heading
        lead = j.reference_xy-position
        j.reference_xy = position + lead*min(1., .10/max(np.linalg.norm(lead), 1e-12))
        j.reference_x = float(heading @ j.reference_xy)
        if (grounded and self.pending_jump and abs(self.yaw_rate) < 1e-6
                and j.request_jump(data, velocity_ref=self.speed)):
            self.pending_jump = False
            print(f'Jump started at {j.c.observe(data)[3]:+.2f} m/s.', flush=True)


def compute_command(controls, data):
    """Run the interactive controller and stop motor commands on invalid/fallen states."""
    jump = controls.jump
    try:
        controls.update(data)
        if controls.righting.active:
            requested = controls.righting.update(data)
            if controls.righting.succeeded:
                # Rebase control references on the recovered pose; never rewrite
                # physical qpos/qvel or call initialize during motor recovery.
                jump.failure = None
                jump.phase, jump.entered = 'RECOVER', float(data.time)
                jump.last_outer = -100.
                jump.reference_xy = np.mean([data.xpos[s['wheel'], :2] for s in jump.c.sides], axis=0)
                jump.reference_x = float(jump.c.observe(data)[0])
                jump.horizontal_velocity_ref = 0.
                controls.target_speed = controls.speed = controls.target_yaw_rate = controls.yaw_rate = 0.
                controls.pending_jump = False
                controls.last_update = float(data.time)
            return requested
        if jump.failure:
            if controls.righting.enabled and not controls.righting.failed:
                if controls.righting.start(data):
                    return controls.righting.update(data)
            return np.zeros(jump.model.nu)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError('Nonfinite state')
        requested = jump.update(data)
        if max(abs(jump.c.body_attitude(data)[:2])) > .8:
            jump.failure = 'Excessive roll/pitch'
    except (ValueError, np.linalg.LinAlgError) as exc:
        jump.failure = str(exc)
        if controls.righting.active:
            controls.righting.fail(str(exc))
    if (jump.failure and controls.righting.enabled and not controls.righting.failed
            and np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
        if controls.step_assist:
            controls.step_assist.cancel()
            jump.set_step_mode(False)
        if controls.righting.start(data):
            controls.target_speed = controls.speed = controls.target_yaw_rate = controls.yaw_rate = 0.
            controls.pending_jump = False
            return controls.righting.update(data)
    return np.zeros(jump.model.nu) if jump.failure else requested


class AutoReset:
    """A simulation reset to spawn, not a motor-driven physical self-righting controller."""

    def __init__(self, enabled=True, delay=1.5):
        self.enabled = enabled
        self.delay = delay
        self.failed_since = None
        self.count = 0

    def update(self, controls, data, now):
        if not controls.jump.failure:
            self.failed_since = None
            return False
        if not self.enabled:
            return False
        if self.failed_since is None:
            self.failed_since = now
        if now-self.failed_since < self.delay:
            return False
        controls.reset(data)
        self.failed_since = None
        self.count += 1
        print(f'Automatic reset #{self.count}: back at initial standing position.', flush=True)
        return True


def create_jump_controller(progress=None, terrain='steps'):
    """Interactive hop: enough flight time to retract and visibly hold the legs."""
    if terrain == 'steps':
        return StepJumpController(interactive=True, progress=progress, model_path=ROOT/'robot_steps.xml')
    return JumpController(auto_jump=False, crouch=.12, extension=.23,
                          thrust_time=.09, boost=50., cut_length=.215,
                          tuck_length=.11, tuck_duration=.055, progress=progress,
                          model_path=ROOT/('robot_steps.xml' if terrain == 'steps' else 'robot.xml'))


def run(args):
    headless = args.smoke_test or args.turn_test
    jump = create_jump_controller(
        progress=lambda L: print(f'Preparing LQR: {L*100:.0f} cm', flush=True), terrain=args.terrain)
    model = jump.model
    configure(model, args.motor_profile)
    data = mujoco.MjData(model)
    controls = KeyboardControl(jump, args.max_speed, self_righting=not args.no_self_righting and not headless)
    controls.reset(data)
    recovery = AutoReset(enabled=args.auto_reset and not headless, delay=args.reset_delay)
    viewer = None
    if not headless:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(model, data, key_callback=controls.keys.put)
        viewer.cam.distance = 1.1
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -12
    print('Focus viewer: UP/DOWN change speed by 0.15 m/s; LEFT/RIGHT change turn rate by 0.5 rad/s; X stop/cancel pending jump; '
          'SPACE ordinary jump; J 15cm step assist; G motor self-righting; R reset; ESC exit. Speed stays set after release.', flush=True)
    if args.step_demo:
        controls.keys.put(ord('J'))
    if controls.righting.enabled:
        print('Falls trigger motor self-righting. Side-lying/inverted poses are outside the validated range.', flush=True)
    if recovery.enabled:
        print(f'Fall/failure: automatic simulation reset after {recovery.delay:g}s; '
              'this optional fallback resets simulation state.', flush=True)
    dt = model.opt.timestep
    # Keep 0.5 ms physics, sample/hold motor commands at the 5 ms controller rate.
    control_steps = jump.c.nsub
    frame_steps = max(1, round(1./(60*dt)))
    tick = 0
    deadline = time.perf_counter()
    run_start = deadline
    previous_phase = jump.phase
    # Exercise the same key queue and control path without opening a window.
    script = iter([(1., KEY_LEFT), (1.1, KEY_LEFT), (1.2, KEY_LEFT), (4.5, ord('X')),
                   (7., KEY_UP), (10., ord(' ')), (17., KEY_RIGHT), (17.1, KEY_RIGHT),
                   (17.2, KEY_RIGHT), (22., ord('X'))] if args.turn_test else
                  [(1., KEY_UP), (4., KEY_DOWN), (7., ord(' ')), (15., ord(' '))])
    event = next(script, None)
    min_x = max_x = float(jump.c.observe(data)[0])
    max_clearance = 0.
    last_yaw = total_yaw = max_yaw = 0.
    motion_start = None
    lateral_travel = 0.
    air_samples = []
    try:
        while (viewer is None or viewer.is_running()):
            if args.duration and tick*dt >= args.duration:
                break
            control_tick = tick % control_steps == 0
            if control_tick:
                mujoco.mj_forward(model, data)
            if headless:
                while event is not None and data.time >= event[0]:
                    controls.keys.put(event[1])
                    event = next(script, None)
            if control_tick:
                previous_failure = jump.failure
                requested = compute_command(controls, data)
                if jump.failure and not controls.righting.active:
                    data.ctrl[:] = 0.
                if jump.failure and previous_failure is None and not controls.righting.active:
                    restart = f'Auto reset in {recovery.delay:g}s.' if recovery.enabled else 'Press R to reset.'
                    print(f'Simulation stopped: {jump.failure}. {restart}', flush=True)
                if not controls.righting.active and recovery.update(controls, data, time.perf_counter()):
                    requested = compute_command(controls, data)
            if jump.phase != previous_phase:
                print(f'{data.time:.2f}s: {jump.phase}', flush=True)
                previous_phase = jump.phase
            if headless and control_tick:
                if jump.failure:
                    raise RuntimeError(jump.failure)
                x = float(jump.c.observe(data)[0])
                min_x, max_x = min(min_x, x), max(max_x, x)
                max_clearance = max(max_clearance, min(jump.last['clearances']))
                if jump.phase in ['FLIGHT', 'LANDING_PREP']:
                    air_samples.append([data.time, *jump.last['lengths'], jump.last['com_vz']])
                yaw = jump.c.body_attitude(data)[2]
                total_yaw += np.arctan2(np.sin(yaw-last_yaw), np.cos(yaw-last_yaw))
                last_yaw = yaw
                max_yaw = max(max_yaw, total_yaw)
                if args.turn_test:
                    xy = np.mean([data.xpos[s['wheel'], :2] for s in jump.c.sides], axis=0)
                    if data.time >= 7. and motion_start is None:
                        motion_start = xy.copy()
                    if 7. <= data.time <= 10.:
                        lateral_travel = max(lateral_travel, abs(xy[1]-motion_start[1]))
            if not jump.failure or controls.righting.active:
                # Reapply the speed-dependent motor envelope at every physics step.
                data.ctrl[:] = limit(model, data, requested, args.motor_profile)
                mujoco.mj_step(model, data)
            tick += 1
            if viewer and tick % frame_steps == 0:
                with viewer.lock():
                    viewer.cam.lookat[:2] = data.xpos[jump.c.chassis, :2]
                    viewer.cam.lookat[2] = .23
                viewer.sync()
                deadline += frame_steps*dt
                time.sleep(max(0., deadline-time.perf_counter()))
                deadline = max(deadline, time.perf_counter()-.1)
    finally:
        if viewer:
            viewer.close()
    elapsed = time.perf_counter()-run_start
    print(f'Simulation throughput: {tick*dt/max(elapsed, 1e-9):.2f}x realtime '
          f'({tick*dt:.2f}s simulated / {elapsed:.2f}s wall)', flush=True)
    if headless:
        flights = sum(e['phase'] == 'FLIGHT' for e in jump.events)
        landings = sum(e['phase'] == 'LANDING' for e in jump.events)
        if args.turn_test:
            assert max_yaw > 1.2 and max_yaw-total_yaw > 1.2, (max_yaw, total_yaw)
            assert lateral_travel > .05, lateral_travel
            assert flights == landings == 1, (flights, landings)
            assert abs(jump.c.body_attitude(data)[5]) < .05
            print(f'Turn check: left={max_yaw:.3f} rad, right={max_yaw-total_yaw:.3f} rad, '
                  f'world-Y travel after turning={lateral_travel:.3f} m')
        else:
            assert max_x-min_x > .05, 'No forward motion'
            assert flights == landings == 2, (flights, landings)
        assert max_clearance > .008, max_clearance
        assert jump.phase == 'RECOVER' and not controls.pending_jump
        assert abs(jump.c.observe(data)[3]) < .05
        samples = np.asarray(air_samples)
        for event in (e for e in jump.events if e['phase'] == 'FLIGHT'):
            end = next(e['time'] for e in jump.events
                       if e['phase'] == 'LANDING' and e['time'] > event['time'])
            air = samples[(samples[:, 0] >= event['time']) & (samples[:, 0] < end)]
            minimum = air[:, 1:3].min(axis=0)
            dwell = float(np.sum(np.max(air[:, 1:3], axis=1) <= jump.tuck_length+.02)*jump.c.dt)
            assert max(minimum) <= jump.tuck_length+.02, minimum
            assert min(air[0, 1:3]-minimum) >= .07, minimum
            assert dwell >= .05, dwell
            acceleration = float(np.polyfit(air[:, 0], air[:, 3], 1)[0])
            assert abs(acceleration+9.81) < .5, acceleration
            print(f'Tuck check: min left/right={minimum*100} cm, '
                  f'compact posture={dwell:.3f}s, COM acceleration={acceleration:.3f} m/s^2')
        print(f'PASS: travel range={max_x-min_x:.3f} m, '
              f'jumps={flights}, landings={landings}, clearance={max_clearance:.3f} m')
        if args.turn_test:
            controls.keys.put(ord('R'))
            controls.update(data)
            assert data.time == 0. and jump.c.yaw_ref == jump.c.yaw_rate_ref == 0.
            assert controls.target_yaw_rate == controls.yaw_rate == 0.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-speed', type=float, default=.4, help='Forward/backward limit, m/s')
    parser.add_argument('--terrain', choices=['steps', 'flat'], default='steps', help='Single 15 cm step or flat floor')
    parser.add_argument('--no-self-righting', action='store_true', help='Disable automatic motor self-righting')
    parser.add_argument('--auto-reset', action='store_true', help='Explicitly enable simulation-reset fallback after failed recovery')
    parser.add_argument('--no-auto-reset', action='store_false', dest='auto_reset', help='Disable simulation-reset fallback (default)')
    parser.set_defaults(auto_reset=False)
    parser.add_argument('--reset-delay', type=float, default=1.5, help='Seconds before automatic reset after failure')
    parser.add_argument('--duration', type=float, default=0., help='Wall-independent simulation duration; 0 = unlimited')
    parser.add_argument('--motor-profile', choices=['xml', 'dm4310-v11-24v', 'dm4310-v11-peak-only'],
                        default='dm4310-v11-24v')
    parser.add_argument('--smoke-test', action='store_true', help='Headless movement and two-jump check')
    parser.add_argument('--turn-test', action='store_true', help='Headless left/right turns, driving after turning and jump check')
    parser.add_argument('--step-demo', action='store_true', help='Launch the viewer and request the J step-assist routine at spawn')
    args = parser.parse_args()
    if not 0 < args.max_speed <= .5 or args.duration < 0:
        parser.error('Require 0 < max-speed <= .5 and duration >= 0')
    if not np.isfinite(args.reset_delay) or args.reset_delay <= 0:
        parser.error('Reset delay must be finite and positive')
    if args.smoke_test and args.turn_test:
        parser.error('Choose either --smoke-test or --turn-test')
    if args.step_demo and (args.terrain != 'steps' or args.smoke_test or args.turn_test):
        parser.error('--step-demo requires the steps terrain and an interactive viewer')
    if (args.smoke_test or args.turn_test) and args.duration == 0:
        args.duration = 27. if args.turn_test else 23.
    run(args)
