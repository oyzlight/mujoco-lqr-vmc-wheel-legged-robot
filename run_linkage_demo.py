"""Bench motion of actual hip inputs; fixed chassis, gravity off, torque PD only."""
import argparse
import json
import time
import mujoco
import numpy as np
from closed_kinematics import ROOT, ClosedLeg


def main(args):
    model=mujoco.MjModel.from_xml_path(str(ROOT/'robot_fixed.xml'))
    model.opt.gravity[:]=0  # A kinematic bench demonstration, not the balance model.
    data=mujoco.MjData(model);leg=ClosedLeg();q=leg.ik(.18,0)
    names=['hip_A','hip_B','output_A','output_B','rocker_pin','rod29_pin','lower_pin','rod35_pin','wheel_joint']
    values=np.r_[-q,leg.forward(q)['angles'],0.]
    for side in ['left','right']:
        for name,value in zip(names,values):
            data.qpos[model.jnt_qposadr[model.joint(side+'_'+name).id]]=value
    mujoco.mj_forward(model,data)
    joints=model.actuator_trnid[:,0];qp=model.jnt_qposadr[joints];qv=model.jnt_dofadr[joints]
    initial=data.qpos[qp].copy()
    viewer=None;wall_start=time.perf_counter();maximum_error=0.
    if args.viewer:
        from mujoco.viewer import launch_passive
        viewer=launch_passive(model,data);viewer.cam.distance=.8
        viewer.cam.lookat[:]=data.subtree_com[model.body('chassis').id]
    try:
        for tick in range(round(args.duration/model.opt.timestep)):
            if viewer and not viewer.is_running():break
            phase=2*np.pi*.4*data.time
            offset=args.amplitude*np.sin(phase)
            rate=args.amplitude*2*np.pi*.4*np.cos(phase)
            signs={'A':(1,0),'B':(0,1),'same':(1,1),'opposite':(1,-1)}[args.case]
            mask=np.array([*signs,0,*signs,0])
            target=initial+offset*mask
            data.ctrl[:]=np.clip(8*(target-data.qpos[qp])+.15*(rate*mask-data.qvel[qv]),model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1])
            mujoco.mj_step(model,data)
            if tick%10==0:
                mujoco.mj_forward(model,data)
                for side in ['left','right']:
                    for p in ['C','D']:
                        maximum_error=max(maximum_error,float(np.linalg.norm(data.site_xpos[model.site(side+'_'+p+'0').id]-data.site_xpos[model.site(side+'_'+p+'1').id])))
                if viewer:
                    viewer.sync();time.sleep(max(0,wall_start+data.time-time.perf_counter()))
        print(json.dumps(dict(case=args.case,duration=data.time,max_closure_error_m=maximum_error,
            finite=bool(np.isfinite(data.qpos).all()),chassis_fixed=True,gravity_enabled=False),indent=2))
        if not np.isfinite(data.qpos).all() or maximum_error>.0005:raise SystemExit(1)
    finally:
        if viewer:viewer.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=['A','B','same','opposite'],default='A')
    p.add_argument('--viewer',action='store_true');p.add_argument('--duration',type=float,default=6.)
    p.add_argument('--amplitude',type=float,default=.12,help='Input motor angular amplitude in radians')
    a=p.parse_args()
    if a.duration<=0 or not 0<a.amplitude<=.2:p.error('Require positive duration and amplitude in (0,.2] rad')
    main(a)
