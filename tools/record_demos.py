"""Record actual motor-driven simulations to MP4/GIF, with checked outcomes."""
import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib.font_manager import findfont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_keyboard import (KeyboardControl, compute_command, create_jump_controller,
                          KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT)
from motor_limits import configure, limit
from tools.validate_self_righting import fallen


def episode(jump, name, events, duration, initial_pitch=None):
    model = jump.model
    data = mujoco.MjData(model)
    keys = KeyboardControl(jump, self_righting=initial_pitch is not None)
    keys.reset(data)
    if initial_pitch is not None:
        # Test initial condition only; all subsequent frames come from physics.
        fallen(jump.c, data, np.deg2rad(initial_pitch))
    events = iter(events)
    event = next(events, None)
    command = np.zeros(model.nu)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.20 if name == 'movement' else 0., 0., .22]
    camera.distance = 1.85 if name == 'movement' else 1.35
    camera.azimuth, camera.elevation = 115., -15.
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    options.sitegroup[4] = 0
    font = ImageFont.truetype(findfont('DejaVu Sans'), 22)
    media = ROOT/'media'
    media.mkdir(exist_ok=True)
    out = media/(name+'.mp4')
    frames_at = 0.
    completed = None
    peak_torque = np.zeros(model.nu)
    xmin = xmax = float(jump.c.observe(data)[0])
    yaw_total = previous_yaw = 0.
    min_yaw = max_yaw = 0.
    with mujoco.Renderer(model, height=544, width=960) as renderer:
        with imageio.get_writer(out, fps=30, codec='libx264', quality=8) as writer:
            for tick in range(round(duration/model.opt.timestep)):
                if tick % jump.c.nsub == 0:
                    mujoco.mj_forward(model, data)
                    while event is not None and data.time >= event[0]:
                        keys.keys.put(event[1])
                        event = next(events, None)
                    before_q, before_v = data.qpos.copy(), data.qvel.copy()
                    command = compute_command(keys, data)
                    assert np.array_equal(before_q, data.qpos) and np.array_equal(before_v, data.qvel)
                    if jump.failure and not keys.righting.active:
                        raise RuntimeError((name, data.time, jump.failure))
                    if keys.righting.failed:
                        raise RuntimeError((name, keys.righting.reason))
                    assert np.all(data.xfrc_applied == 0) and np.all(data.qfrc_applied == 0)
                    state = jump.c.observe(data)
                    xmin, xmax = min(xmin, state[0]), max(xmax, state[0])
                    yaw = float(jump.c.body_attitude(data)[2])
                    yaw_total += float(np.arctan2(np.sin(yaw-previous_yaw), np.cos(yaw-previous_yaw)))
                    previous_yaw = yaw
                    min_yaw, max_yaw = min(min_yaw, yaw_total), max(max_yaw, yaw_total)
                    if keys.righting.succeeded and completed is None:
                        completed = float(data.time)
                if data.time+1e-9 >= frames_at:
                    mujoco.mj_forward(model, data)
                    if name == 'step15':
                        camera.lookat[:] = [data.qpos[0]+.25, data.qpos[1], .23]
                        camera.distance = 1.45
                    renderer.update_scene(data, camera=camera, scene_option=options)
                    frame = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(frame)
                    phase = keys.righting.phase if keys.righting.active or keys.righting.succeeded else jump.phase
                    draw.rectangle((0, 0, 960, 68), fill=(22, 30, 42))
                    title = name.replace('_', ' ').upper()
                    if initial_pitch is not None:
                        title += f' | initial pitch {initial_pitch:+g} deg'
                    draw.text((15, 5), f'{title} | {phase} | t={data.time:.2f} s', font=font, fill='white')
                    draw.text((15, 35), 'MuJoCo LQR + VMC | six motor torques | original 24V torque-speed limits',
                              font=font, fill='#c7d5e7')
                    writer.append_data(np.asarray(frame))
                    frames_at += 1/30
                data.ctrl[:] = limit(model, data, command, 'dm4310-v11-24v')
                peak_torque = np.maximum(peak_torque, abs(data.ctrl))
                assert np.all(peak_torque <= model.actuator_ctrlrange[:, 1]+1e-9)
                mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    info = jump.contact_info(data)
    tilt = max(abs(jump.c.body_attitude(data)[:2]))
    assert min(info['forces']) > 3. and not info['nonwheel'] and tilt < .05
    assert abs(jump.c.observe(data)[3]) < .05
    if name == 'movement':
        assert xmax-xmin > .5
    if name == 'rotation':
        assert max_yaw > 1. and max_yaw-yaw_total > 1.
    if name == 'jump':
        assert any(e['phase'] == 'FLIGHT' for e in jump.events)
        assert jump.phase == 'RECOVER'
    if initial_pitch is not None:
        assert keys.righting.succeeded and completed is not None and data.qpos[2] > .20
    if name == 'step15':
        assert keys.step_assist.phase == 'DONE'
        assert min(info['wheel_positions'][:, 0]) > 1.55
        assert max(abs(info['wheel_positions'][:, 2]-.19)) < .005 and data.qpos[2] > .35
    report = dict(video=out.name, passed=True, simulated_duration_s=float(data.time),
                  peak_motor_torque_Nm=peak_torque.tolist(), final_body_height_m=float(data.qpos[2]),
                  final_wheel_positions_m=info['wheel_positions'].tolist(),
                  final_speed_m_s=float(jump.c.observe(data)[3]), final_tilt_rad=float(tilt),
                  initial_pitch_deg=initial_pitch, physical_recovery_complete_s=completed,
                  state_rewrites_during_control=False, external_forces=False)
    print(json.dumps(report), flush=True)
    return report


def gif_preview(path):
    # Preview only; MP4 contains the complete normal-speed episode.
    frames = []
    with imageio.get_reader(path) as reader:
        for i, array in enumerate(reader):
            if i % 4:
                continue
            frame = Image.fromarray(array).resize((360, 204), Image.Resampling.LANCZOS)
            frames.append(frame.convert('P', palette=Image.Palette.ADAPTIVE, colors=128))
    frames[0].save(path.with_suffix('.gif'), save_all=True, append_images=frames[1:],
                   duration=133, loop=0, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['all', 'movement', 'rotation', 'jump', 'self_righting', 'step15'], default='all')
    args = parser.parse_args()
    reports = []
    if args.case != 'step15':
        jump = create_jump_controller(terrain='flat')
        configure(jump.model, 'dm4310-v11-24v')
        scripts = {
            'movement': ([(.5, KEY_UP), (.51, KEY_UP), (4., ord('X')),
                          (6.5, KEY_DOWN), (6.51, KEY_DOWN), (10., ord('X'))], 13.5),
            'rotation': ([(1., KEY_LEFT), (1.1, KEY_LEFT), (1.2, KEY_LEFT), (4., ord('X')),
                          (6., KEY_RIGHT), (6.1, KEY_RIGHT), (6.2, KEY_RIGHT), (10., ord('X'))], 13.5),
            'jump': ([(.75, ord(' '))], 5.5),
        }
        for name, (events, duration) in scripts.items():
            if args.case in ['all', name]:
                reports.append(episode(jump, name, events, duration))
        if args.case in ['all', 'self_righting']:
            for angle in [70., -70.]:
                reports.append(episode(jump, 'self_righting_'+('forward' if angle > 0 else 'backward'), [], 5.5, angle))
    if args.case in ['all', 'step15']:
        jump = create_jump_controller(terrain='steps')
        configure(jump.model, 'dm4310-v11-24v')
        reports.append(episode(jump, 'step15', [(1., ord('J'))], 10.))
    for report in reports:
        gif_preview(ROOT/'media'/report['video'])
    (ROOT/'media/demo_report.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
