"""Validate motor-only recovery, including contact support and no pose rewrites."""
from pathlib import Path
import sys
import json

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_keyboard import KeyboardControl, compute_command, create_jump_controller
from motor_limits import configure, limit


def fallen(c, data, pitch, roll=0.):
    """Set a fallen test initial condition with collision proxies above the floor."""
    c.initialize(data)
    quat = Rotation.from_euler('xyz', [roll, pitch, 0.]).as_quat()
    data.qpos[3:7] = quat[[3, 0, 1, 2]]
    mujoco.mj_forward(c.model, data)
    minimum = np.inf
    for g in range(c.model.ngeom):
        if c.model.geom_bodyid[g] == 0 or not c.model.geom_contype[g]:
            continue
        R, size = data.geom_xmat[g].reshape(3, 3), c.model.geom_size[g]
        kind = c.model.geom_type[g]
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            extent = abs(R[2]) @ size
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            extent = size[0]+abs(R[2, 2])*size[1]
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            extent = size[0]*np.linalg.norm(R[2, :2])+size[1]*abs(R[2, 2])
        else:
            continue
        minimum = min(minimum, data.geom_xpos[g, 2]-extent)
    data.qpos[2] += .001-minimum
    mujoco.mj_forward(c.model, data)


def main():
    jump = create_jump_controller(terrain='flat')
    model = jump.model
    configure(model, 'dm4310-v11-24v')
    data = mujoco.MjData(model)
    keys = KeyboardControl(jump)
    keys.reset(data)
    initial = data.qpos.copy()
    keys.keys.put(ord('G'))
    compute_command(keys, data)
    assert not keys.righting.active and not jump.failure
    assert np.array_equal(initial, data.qpos)
    reports = []
    for degrees in [70., -70., 110., -110.]:
        keys.reset(data)
        fallen(jump.c, data, np.deg2rad(degrees))
        rows = []
        peak = np.zeros(model.nu)
        command = np.zeros(model.nu)
        completed = None
        for tick in range(round(12/model.opt.timestep)):
            if tick % jump.c.nsub == 0:
                mujoco.mj_forward(model, data)
                before_q, before_v = data.qpos.copy(), data.qvel.copy()
                command = compute_command(keys, data)
                assert np.array_equal(before_q, data.qpos), 'Recovery wrote qpos'
                assert np.array_equal(before_v, data.qvel), 'Recovery wrote qvel'
                assert not keys.righting.failed, (degrees, keys.righting.reason, data.time)
                assert np.all(data.xfrc_applied == 0) and np.all(data.qfrc_applied == 0)
                info = jump.contact_info(data)
                attitude = jump.c.body_attitude(data)
                rows.append([data.time, *attitude, *info['forces'], int(info['nonwheel']),
                             data.qpos[2], *data.qpos])
                if keys.righting.succeeded and completed is None:
                    completed = float(data.time)
                if completed is not None and data.time-completed >= 1.5:
                    assert not jump.failure, jump.failure
                    assert min(info['forces']) > 3. and not info['nonwheel']
                    assert max(abs(attitude[:2])) < .03
                    assert abs(jump.c.observe(data)[3]) < .05
                    assert data.qpos[2] > .20
                    break
            data.ctrl[:] = limit(model, data, command, 'dm4310-v11-24v')
            peak = np.maximum(peak, abs(data.ctrl))
            assert np.all(peak <= model.actuator_ctrlrange[:, 1]+1e-9)
            mujoco.mj_step(model, data)
        assert completed is not None, (degrees, keys.righting.phase, keys.righting.reason)
        path = ROOT/f'results/self_righting_{degrees:g}.csv'
        np.savetxt(path, rows, delimiter=',')
        report = dict(initial_pitch_deg=degrees, completion_time_s=completed,
                      final_height_m=float(data.qpos[2]), final_pitch_deg=float(np.rad2deg(attitude[1])),
                      peak_motor_torque_Nm=peak.tolist(), attempts=keys.righting.attempts,
                      both_wheels_supporting=True, body_ground_contact=False,
                      state_rewrites_during_control=False, external_forces=False)
        reports.append(report)
        print('PASS', report, flush=True)
    for pitch, roll in [(0., 70.), (170., 0.)]:
        keys.reset(data)
        fallen(jump.c, data, np.deg2rad(pitch), np.deg2rad(roll))
        before = data.qpos.copy()
        compute_command(keys, data)
        assert keys.righting.failed and not keys.righting.succeeded
        assert np.array_equal(before, data.qpos)
        print('Unsupported pose correctly retained:', pitch, roll, keys.righting.reason, flush=True)
    output = dict(passed=True, supported_front_back_cases=reports,
                  side_lying_supported=False, fully_inverted_supported=False,
                  simulation_reset_used=False)
    (ROOT/'results/self_righting_validation.json').write_text(json.dumps(output, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
