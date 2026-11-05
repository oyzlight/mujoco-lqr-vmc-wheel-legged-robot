"""End-to-end J-key validation: land and stay balanced on the unchanged 15 cm step."""
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_keyboard import create_jump_controller, KeyboardControl, compute_command
from motor_limits import configure, limit


def main():
    jump = create_jump_controller()
    model = jump.model
    configure(model, 'dm4310-v11-24v')
    data = mujoco.MjData(model)
    keys = KeyboardControl(jump, self_righting=False)
    reports = []
    for initial_x in [0., -.02, .02]:
        keys.reset(data)
        data.qpos[0] += initial_x
        mujoco.mj_forward(model, data)
        jump.reference_xy[0] += initial_x
        jump.reference_x = float(jump.c.observe(data)[0])
        command = np.zeros(model.nu)
        rows = []
        peak_torque = np.zeros(model.nu)
        peak_clearance = 0.
        maximum_loop = maximum_gear = 0.
        for tick in range(round(12/model.opt.timestep)):
            if tick % jump.c.nsub == 0:
                mujoco.mj_forward(model, data)
                if tick == round(1/model.opt.timestep):
                    keys.keys.put(ord('J'))
                before_q, before_v = data.qpos.copy(), data.qvel.copy()
                command = compute_command(keys, data)
                assert np.array_equal(before_q, data.qpos) and np.array_equal(before_v, data.qvel)
                assert not jump.failure, (initial_x, jump.failure, data.time)
                assert not keys.righting.active
                assert np.all(data.xfrc_applied == 0) and np.all(data.qfrc_applied == 0)
                info = jump.contact_info(data)
                peak_clearance = max(peak_clearance, min(info['wheel_positions'][:, 2])-jump.c.radius)
                if tick % (4*jump.c.nsub) == 0:
                    loop, gear = jump.c.constraint_errors(data)
                    maximum_loop, maximum_gear = max(maximum_loop, loop), max(maximum_gear, gear)
                rows.append([data.time, jump.PHASES.index(jump.phase),
                             *np.mean(info['wheel_positions'], axis=0),
                             *jump.last['lengths'], *info['forces'],
                             *jump.c.body_attitude(data)[:3], *data.qpos, *data.qvel])
            data.ctrl[:] = limit(model, data, command, 'dm4310-v11-24v')
            peak_torque = np.maximum(peak_torque, abs(data.ctrl))
            assert np.all(peak_torque <= model.actuator_ctrlrange[:, 1]+1e-9)
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        info = jump.contact_info(data)
        att = jump.c.body_attitude(data)
        speed = float(jump.c.observe(data)[3])
        assert keys.step_assist.phase == 'DONE' and jump.phase == 'RECOVER'
        assert sum(e['phase'] == 'FLIGHT' for e in jump.events) == 1
        assert sum(e['phase'] == 'LANDING' for e in jump.events) == 1
        assert min(info['wheel_positions'][:, 0]) > 1.55
        assert max(abs(info['wheel_positions'][:, 2]-.19)) < .005
        assert min(info['forces']) > 3. and not info['nonwheel']
        assert max(abs(att[:2])) < .03 and abs(speed) < .03
        assert data.qpos[2] > .35 and maximum_loop < .0005 and maximum_gear < .01
        report = dict(initial_x_offset_m=initial_x, passed=True, final_body_height_m=float(data.qpos[2]),
                      final_wheel_positions_m=info['wheel_positions'].tolist(), final_speed_m_s=speed,
                      max_wheel_bottom_m=float(peak_clearance), final_attitude_rad=att[:3].tolist(),
                      peak_motor_torques_Nm=peak_torque.tolist(), max_loop_error_m=float(maximum_loop),
                      max_gear_error_rad=float(maximum_gear), events=jump.events,
                      external_forces=False, state_resets_during_control=False, motor_self_righting_used=False)
        reports.append(report)
        if initial_x == 0.:
            np.savetxt(ROOT/'results/step15_trajectory.csv', rows, delimiter=',')
        print('PASS', json.dumps(report), flush=True)
    (ROOT/'results/step15_validation.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
