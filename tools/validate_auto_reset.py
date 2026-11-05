"""Validate the 15 cm scene and the interactive failure/reset/resume path."""
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_keyboard import AutoReset, KeyboardControl, KEY_UP, compute_command, create_jump_controller
from motor_limits import configure, limit


def main():
    jump = create_jump_controller()
    model = jump.model
    steps = [g for g in range(model.ngeom) if model.geom(g).name.startswith('step_')]
    assert len(steps) == 1
    step = steps[0]
    assert np.isclose(model.geom_pos[step, 2]+model.geom_size[step, 2], .15)
    assert model.geom_contype[step] != 0 and model.geom_conaffinity[step] != 0
    data = mujoco.MjData(model)
    configure(model, 'dm4310-v11-24v')
    controls = KeyboardControl(jump, self_righting=False)
    controls.reset(data)
    initial = data.qpos.copy()
    recovery = AutoReset(delay=1.5)

    # Rotate the actual floating base beyond the fall threshold.
    data.qpos[0] += .3
    data.qpos[3:7] = [np.cos(.6), np.sin(.6), 0., 0.]
    mujoco.mj_forward(model, data)
    command = compute_command(controls, data)
    assert jump.failure and np.all(command == 0)
    assert not recovery.update(controls, data, 100.)
    assert not recovery.update(controls, data, 101.49)
    controls.keys.put(KEY_UP)
    controls.keys.put(ord(' '))
    assert recovery.update(controls, data, 101.5)
    assert np.allclose(data.qpos, initial, atol=1e-10) and np.all(data.qvel == 0)
    assert data.time == 0 and jump.failure is None and jump.phase == 'BALANCE'
    assert controls.speed == controls.target_speed == controls.yaw_rate == controls.target_yaw_rate == 0
    assert not controls.pending_jump and controls.keys.empty()

    # Confirm that a restored robot remains balanced and accepts new arrow input.
    maximum_speed = 0.
    for tick in range(round(3/model.opt.timestep)):
        if tick % jump.c.nsub == 0:
            mujoco.mj_forward(model, data)
            if tick == round(1/model.opt.timestep):
                controls.keys.put(KEY_UP)
            command = compute_command(controls, data)
            assert not jump.failure, jump.failure
            assert not recovery.update(controls, data, 102.+data.time)
            maximum_speed = max(maximum_speed, float(jump.c.observe(data)[3]))
        data.ctrl[:] = limit(model, data, command, 'dm4310-v11-24v')
        mujoco.mj_step(model, data)
    assert maximum_speed > .1

    # Nonfinite states also reset, and manual/disabled modes keep their semantics.
    data.qvel[0] = np.nan
    assert np.all(compute_command(controls, data) == 0) and jump.failure
    assert not recovery.update(controls, data, 110.)
    assert recovery.update(controls, data, 111.5) and recovery.count == 2
    jump.failure = 'Manual reset test'
    assert not recovery.update(controls, data, 120.)
    controls.keys.put(ord('R'))
    compute_command(controls, data)
    assert jump.failure is None and not recovery.update(controls, data, 122.)
    jump.failure = 'Disabled reset test'
    disabled = AutoReset(enabled=False)
    assert not disabled.update(controls, data, 0.)
    assert not disabled.update(controls, data, 100.) and jump.failure
    controls.reset(data)

    report = dict(passed=True, step_count=1, step_height_m=.15, auto_reset_count=recovery.count,
                  reset_delay_s=recovery.delay, fall_detection=True, nonfinite_detection=True,
                  stale_keys_cleared=True, resumed_driving=True, manual_reset=True,
                  disabled_mode=True, physical_self_righting_enabled_in_this_test=False,
                  step_jump_validated=False)
    output = ROOT/'results/auto_reset_validation.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
