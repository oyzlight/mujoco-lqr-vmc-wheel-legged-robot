"""Exercise arrow/space commands for moving takeoff on the flat-floor fixture."""
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_keyboard import create_jump_controller, KeyboardControl, KEY_UP, KEY_DOWN
from motor_limits import configure, limit


def main():
    jump = create_jump_controller(terrain='flat')
    model = jump.model
    configure(model, 'dm4310-v11-24v')
    data = mujoco.MjData(model)
    keys = KeyboardControl(jump)
    reports = []
    for case, direction in [('forward', 1), ('backward', -1)]:
        keys.reset(data)
        for _ in range(2):
            keys.keys.put(KEY_UP if direction > 0 else KEY_DOWN)
        requested = np.zeros(model.nu)
        queued = 0
        stopped = False
        takeoff = landing = None
        rows = []
        peak_tilt = 0.
        for tick in range(round(10/model.opt.timestep)):
            if tick % jump.c.nsub == 0:
                mujoco.mj_forward(model, data)
                xy = np.mean([data.xpos[s['wheel'], :2] for s in jump.c.sides], axis=0)
                trigger = not queued and data.time >= 1.5
                if trigger:
                    keys.keys.put(ord(' ')); queued += 1
                if jump.phase == 'LANDING' and not stopped:
                    keys.keys.put(ord('X')); stopped = True
                keys.update(data)
                previous = jump.phase
                requested = jump.update(data)
                state = jump.c.observe(data)
                attitude = jump.c.body_attitude(data)
                peak_tilt = max(peak_tilt, float(max(abs(attitude[:2]))))
                if jump.phase != previous:
                    print(case, round(data.time, 3), jump.phase, 'x', round(xy[0], 3),
                          'vx', round(state[3], 3), flush=True)
                if jump.phase == 'FLIGHT' and takeoff is None:
                    takeoff = dict(x=float(xy[0]), vx=float(state[3]))
                if jump.phase == 'LANDING' and landing is None:
                    landing = dict(x=float(xy[0]), vx=float(state[3]),
                                   wheel_z=[float(data.xpos[s['wheel'], 2]) for s in jump.c.sides])
                rows.append([data.time, jump.PHASES.index(jump.phase), *xy,
                             *jump.last['lengths'], *jump.last['forces'],
                             *attitude[:3], *data.qpos])
                if jump.failure:
                    raise AssertionError((case, jump.failure, data.time, xy))
                assert peak_tilt < .3, (case, peak_tilt)
            data.ctrl[:] = limit(model, data, requested, 'dm4310-v11-24v')
            mujoco.mj_step(model, data)
        assert takeoff and landing, (case, takeoff, landing)
        assert direction*takeoff['vx'] > .15, (case, takeoff)
        assert direction*(landing['x']-takeoff['x']) > .025, (case, takeoff, landing)
        assert jump.phase == 'RECOVER' and abs(jump.c.observe(data)[3]) < .05
        report = dict(case=case, takeoff=takeoff, landing=landing,
                      airborne_travel_m=landing['x']-takeoff['x'], peak_tilt_deg=float(np.rad2deg(peak_tilt)),
                      landings=sum(e['phase'] == 'LANDING' for e in jump.events))
        reports.append(report)
        np.savetxt(ROOT/f'results/moving_jump_{case}.csv', rows, delimiter=',')
        print('PASS', report, flush=True)
    (ROOT/'results/moving_jump_validation.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
