"""Evaluate controller-only candidates on the unchanged 15 cm collision model."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import mujoco
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from step_jump import StepJumpController
from run_keyboard import KeyboardControl
from motor_limits import configure, limit


def evaluate(jump, speed, trigger, offset=0., save=False):
    model = jump.model
    data = mujoco.MjData(model)
    keys = KeyboardControl(jump, max_speed=1., self_righting=False)
    keys.reset(data)
    keys.target_speed = speed
    jump.foot_offset = offset
    queued = False
    stopping = False
    command = np.zeros(model.nu)
    peak = 0.
    rows = []
    outcome = 'timeout'
    for tick in range(round(9/model.opt.timestep)):
        if tick % jump.c.nsub == 0:
            mujoco.mj_forward(model, data)
            x = float(np.mean([data.xpos[s['wheel'], 0] for s in jump.c.sides]))
            if x >= trigger and not queued:
                keys.keys.put(ord(' ')); queued = True
            if jump.phase == 'LANDING' and not stopping:
                keys.target_speed = 0.; stopping = True
            keys.update(data)
            try:
                command = jump.update(data)
            except (ValueError, np.linalg.LinAlgError) as exc:
                outcome = str(exc); break
            peak = max(peak, min(data.xpos[s['wheel'], 2]-jump.c.radius for s in jump.c.sides))
            att = jump.c.body_attitude(data)
            rows.append([data.time, jump.PHASES.index(jump.phase), x,
                         *jump.last['lengths'], *jump.last['forces'], *att[:3], *data.qpos, *data.qvel])
            if jump.failure:
                outcome = jump.failure; break
            if max(abs(att[:2])) > .65:
                outcome = 'Excessive attitude'; break
            if jump.phase == 'RECOVER' and data.time-jump.entered > 3.:
                z = [float(data.xpos[s['wheel'], 2]) for s in jump.c.sides]
                good = (1.55 < x < 3.2 and max(abs(np.asarray(z)-.19)) < .005
                        and min(jump.last['forces']) > 3. and abs(jump.c.observe(data)[3]) < .05
                        and max(abs(att[:2])) < .05 and data.qpos[2] > .34)
                outcome = 'PASS' if good else 'Unstable or missed platform'
                break
        data.ctrl[:] = limit(model, data, command, 'dm4310-v11-24v')
        mujoco.mj_step(model, data)
    report = dict(speed=speed, trigger=trigger, offset=offset, outcome=outcome,
                  time=float(data.time), x=float(data.qpos[0]), height=float(data.qpos[2]),
                  velocity=float(jump.c.observe(data)[3]), attitude=jump.c.body_attitude(data).tolist(),
                  peak_wheel_bottom_m=float(peak), phases=jump.events)
    if save:
        np.savetxt(ROOT/'results/step_jump_candidate.csv', rows, delimiter=',')
        (ROOT/'results/step_jump_candidate.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--speeds', default='.6,.8')
    p.add_argument('--triggers', default='.5,.6,.7')
    p.add_argument('--offsets', default='0')
    p.add_argument('--all', action='store_true')
    args = p.parse_args()
    jump = StepJumpController(model_path=ROOT/'robot_steps.xml',
                              progress=lambda L: print('Preparing', L, flush=True))
    configure(jump.model, 'dm4310-v11-24v')
    reports = []
    for speed in map(float, args.speeds.split(',')):
        for trigger in map(float, args.triggers.split(',')):
            for offset in map(float, args.offsets.split(',')):
                report = evaluate(jump, speed, trigger, offset)
                reports.append(report)
                print(json.dumps(report), flush=True)
                if report['outcome'] == 'PASS':
                    if not args.all:
                        evaluate(jump, speed, trigger, offset, save=True)
                        (ROOT/'results/step_jump_search.json').write_text(json.dumps(reports, indent=2))
                        return
    (ROOT/'results/step_jump_search.json').write_text(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
