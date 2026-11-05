"""Replay recorded jump qpos to MP4; this script does not alter the simulation."""
import argparse
import json

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib.font_manager import findfont

from lqr_vmc import ROOT


def main(args):
    directory=ROOT/args.input
    a=np.genfromtxt(directory/'trajectory.csv',delimiter=',',names=True)
    parameters=json.loads((directory/'parameters.json').read_text())
    phases=parameters['phases']
    profile=parameters.get('motor_profile','xml')
    actions=parameters.get('air_actions',[])
    model=mujoco.MjModel.from_xml_path(str(ROOT/'robot.xml'))
    data=mujoco.MjData(model)
    qpos=np.column_stack([a[f'qpos_{i}'] for i in range(model.nq)])
    camera=mujoco.MjvCamera()
    camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:]=[0,0,.18]
    camera.distance=.78
    camera.azimuth=120
    camera.elevation=-14
    option=mujoco.MjvOption()
    option.geomgroup[3]=0
    font=ImageFont.truetype(findfont('DejaVu Sans'),22)
    small=ImageFont.truetype(findfont('DejaVu Sans'),17)
    out=directory/'jump.mp4'
    # Slow motion around the actual hop, normal speed while standing/crouching.
    sample_times=[]
    t=0.
    while t<a['time'][-1]:
        sample_times.append(t)
        t+=(args.slow_motion if 2.8<t<3.9 else 1.)/args.fps
    apex=int(np.argmax(a['com_z']))
    air_indices=np.flatnonzero(np.isin(a['phase'],[3,4]))
    tucked=(int(air_indices[np.argmin((a['left_length'][air_indices]+a['right_length'][air_indices])/2)])
             if len(air_indices) else apex)
    with mujoco.Renderer(model,height=640,width=960) as renderer:
        with imageio.get_writer(out,fps=args.fps,codec='libx264',quality=8) as writer:
            for t in sample_times:
                idx=int(np.clip(np.searchsorted(a['time'],t),0,len(a)-1))
                data.qpos[:]=qpos[idx]
                mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera=camera,scene_option=option)
                frame=Image.fromarray(renderer.render())
                draw=ImageDraw.Draw(frame)
                phase=phases[int(a['phase'][idx])]
                if actions and 'air_action' in a.dtype.names:
                    action=actions[int(a['air_action'][idx])]
                    if action!='NONE':phase+=f' / {action}'
                draw.rectangle((0,0,960,130),fill=(22,30,42))
                draw.text((20,10),f'Physical jump | {phase} | t={a["time"][idx]:.2f} s',font=font,fill='white')
                clearance=min(a['left_clearance'][idx],a['right_clearance'][idx])
                draw.text((20,45),f'Both-wheel clearance: {clearance*100:.1f} cm   COM z: {a["com_z"][idx]:.3f} m',font=small,fill='#c7d5e7')
                length=(a['left_length'][idx]+a['right_length'][idx])/2
                speed=f'{args.slow_motion:g}x slow motion' if 2.8<t<3.9 else 'normal speed'
                draw.text((20,72),f'Leg: {length*100:.1f} cm | target: {a["length_ref"][idx]*100:.1f} cm | {speed}',
                          font=small,fill='#c7d5e7')
                draw.text((20,98),{'xml':'Motor model: original constant torque limits',
                                  'dm4310-v11-24v':'Motor model: APPROXIMATE 24V torque-speed envelope',
                                  'dm4310-v11-peak-only':'Motor model: IDEAL peak torque, speed derating omitted'}[profile],
                          font=small,fill='#ffd37a')
                writer.append_data(np.asarray(frame))
                if abs(idx-apex)<=3:
                    frame.save(directory/'apex.png')
                if abs(idx-tucked)<=3:
                    frame.save(directory/'most_tucked.png')
    print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='results/closed_loop/jump')
    p.add_argument('--fps',type=int,default=30)
    p.add_argument('--slow-motion',type=float,default=.2)
    args=p.parse_args()
    if args.fps<=0:p.error('FPS must be positive')
    if not 0<args.slow_motion<=1:p.error('Slow motion rate must be in (0,1]')
    main(args)
