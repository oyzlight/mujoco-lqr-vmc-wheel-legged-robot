"""Single physical jump: crouch, thrust, flight, landing and LQR recovery."""
import argparse
import csv
import json
import time
import sys

import mujoco
import numpy as np

from md.jump_controller import JumpController,CONTROLLER_VERSION
from lqr_vmc import ROOT
from motor_limits import configure as configure_motors, limit as limit_motors


def run(args):
    jump=JumpController(crouch=args.crouch,extension=args.extension,
                        thrust_time=args.thrust_time,boost=args.boost,cut_length=args.cut_length,
                        tuck_length=None if args.no_tuck else args.tuck_length,force_limit=args.force_limit,
                        deep_schedule=args.deep_schedule,wheel_contact_time=args.wheel_contact_time,
                        progress=(lambda L:print(f'Preparing balance/LQR at leg length {100*L:.0f} cm...',file=sys.stderr,flush=True)) if args.viewer else None)
    if args.takeoff_momentum_gain is not None:
        jump.takeoff_momentum_gain=args.takeoff_momentum_gain
    if args.takeoff_momentum_target is not None:
        jump.takeoff_momentum_target=args.takeoff_momentum_target
    model=jump.model
    configure_motors(model,args.motor_profile)
    data=mujoco.MjData(model)
    jump.initialize(data)
    output=ROOT/args.output
    output.mkdir(parents=True,exist_ok=True)
    viewer=None
    if args.viewer:
        from mujoco import viewer as mjviewer
        viewer=mjviewer.launch_passive(model,data)
        viewer.cam.distance=.95;viewer.cam.azimuth=120;viewer.cam.elevation=-10
    rows=[]; saturation=0; steps=0
    maximum_speeds=np.zeros(model.nu)
    maximum_torques=np.zeros(model.nu)
    initial_com=float(data.subtree_com[jump.c.chassis,2])
    dt=model.opt.timestep
    wall_start=time.perf_counter()
    try:
        for tick in range(round(args.duration/dt)):
            start=time.perf_counter()
            if viewer and not viewer.is_running():
                jump.failure='Viewer closed early';break
            mujoco.mj_forward(model,data)
            requested=jump.update(data)
            data.ctrl[:]=limit_motors(model,data,requested,args.motor_profile)
            maximum_torques=np.maximum(maximum_torques,abs(data.ctrl))
            saturation+=int(jump.last['saturation'] or np.any(abs(requested-data.ctrl)>1e-9));steps+=1
            for actuator in range(model.nu):
                joint=model.actuator_trnid[actuator,0]
                maximum_speeds[actuator]=max(maximum_speeds[actuator],abs(data.qvel[model.jnt_dofadr[joint]]))
            if tick%5==0:
                info=jump.last
                roll,pitch,yaw,*_=jump.c.body_attitude(data)
                state=jump.c.observe(data)
                rows.append([data.time,jump.PHASES.index(jump.phase),data.qpos[2],
                             info['com_z'],info['com_vz'],*info['clearances'],*info['forces'],
                             *info['lengths'],info['length_ref'],roll,pitch,yaw,
                             state[0],state[3],*data.ctrl,int(info['saturation']),*data.qpos,
                             jump.AIR_ACTIONS.index(jump.air_action)])
            if jump.failure:
                break
            attitude=jump.c.body_attitude(data)
            if max(abs(attitude[:2]))>.8 or not np.isfinite(data.qpos).all():
                jump.failure='Nonfinite state or roll/pitch exceeds 0.8 rad';break
            mujoco.mj_step(model,data)
            if viewer:
                if tick%10==0:
                    viewer.cam.lookat[:]=[0,0,.23]
                    viewer.sync()
                    time.sleep(max(0,wall_start+data.time-time.perf_counter()))
    finally:
        if viewer:viewer.close()
    if not rows:raise RuntimeError('No samples')
    a=np.asarray(rows)
    header=['time','phase','base_z','com_z','com_vz','left_clearance','right_clearance',
            'left_normal_force','right_normal_force','left_length','right_length','length_ref',
            'roll','pitch','yaw','axle_x','axle_vx',
            'left_hip_A_torque','left_hip_B_torque','left_wheel_torque',
            'right_hip_A_torque','right_hip_B_torque','right_wheel_torque','saturated']
    header += [f'qpos_{i}' for i in range(model.nq)]
    header += ['air_action']
    with (output/'trajectory.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(header);writer.writerows(rows)
    tail=a[a[:,0]>=max(0,a[-1,0]-1)]
    landed=a[a[:,1]>=jump.PHASES.index('LANDING')]
    post_landing_max_position_error=(float(np.max(abs(landed[:,15]-jump.reference_x))) if len(landed) else None)
    post_landing_max_speed=(float(np.max(abs(landed[:,16]))) if len(landed) else None)
    post_landing_travel=(float(np.sum(abs(np.diff(landed[:,15])))) if len(landed) else None)
    airborne=a[(a[:,1]>=jump.PHASES.index('FLIGHT'))
               & (a[:,1]<=jump.PHASES.index('LANDING_PREP'))
               & (np.max(a[:,7:9],axis=1)<.1)]
    ballistic_acceleration=(float(np.polyfit(airborne[:,0],airborne[:,4],1)[0])
                            if len(airborne)>=5 else None)
    minimum_air_length=float(np.min(np.mean(airborne[:,9:11],axis=1))) if len(airborne) else None
    tuck_stroke=float(np.mean(airborne[0,9:11])-minimum_air_length) if len(airborne) else None
    tuck_achieved=bool(jump.tuck_length is not None and minimum_air_length is not None
                       and minimum_air_length<=jump.tuck_length+.02 and tuck_stroke>=.04)
    report=dict(model='robot.xml',duration_completed=float(data.time),failure=jump.failure,
                controller_version=CONTROLLER_VERSION,
                motor_profile=args.motor_profile,
                events=jump.events,confirmed_takeoff=jump.flight_start is not None,
                confirmed_landing=jump.landing_time is not None,
                flight_time_s=None if jump.landing_time is None else jump.landing_time-jump.flight_start,
                max_two_wheel_clearance_m=float(np.max(np.min(a[:,5:7],axis=1))),
                initial_com_height_m=initial_com,com_rise_from_standing_m=float(max(a[:,3])-initial_com),
                flight_com_rise_m=None if jump.flight_start is None else float(max(a[:,3])-jump.flight_com_z),
                takeoff_com_velocity_m_s=jump.flight_vz,
                takeoff_pitch_momentum_Nms=jump.flight_pitch_momentum,
                air_actions=jump.air_events,tuck_target_m=jump.tuck_length,
                minimum_air_leg_length_m=minimum_air_length,tuck_retraction_stroke_m=tuck_stroke,
                tuck_achieved=tuck_achieved,
                takeoff_momentum_target_Nms=jump.takeoff_momentum_target,
                tail_pitch_rms_deg=float(np.rad2deg(np.sqrt(np.mean(tail[:,13]**2)))),
                tail_roll_rms_deg=float(np.rad2deg(np.sqrt(np.mean(tail[:,12]**2)))),
                tail_yaw_rms_deg=float(np.rad2deg(np.sqrt(np.mean(tail[:,14]**2)))),
                final_position_error_m=float(a[-1,15]-jump.reference_x),
                post_landing_max_position_error_m=post_landing_max_position_error,
                post_landing_max_speed_m_s=post_landing_max_speed,
                post_landing_total_travel_m=post_landing_travel,
                tail_length_error_m=float(np.max(abs(tail[:,9:11]-.18))),
                final_forward_speed_m_s=float(a[-1,16]),
                maximum_leg_length_m=float(np.max(a[:,9:11])),
                peak_pitch_deg=float(np.rad2deg(np.max(abs(a[:,13])))),
                airborne_com_acceleration_m_s2=ballistic_acceleration,
                maximum_motor_torque_Nm=maximum_torques.tolist(),
                torque_statistics_period_s=float(dt),wheel_contact_time=args.wheel_contact_time,
                final_contact_solref_times_s=[float(c.solref[0]) for c in data.contact],
                maximum_motor_speed_rad_s=maximum_speeds.tolist(),
                saturation_fraction=saturation/max(1,steps),
                external_forces_used=False,root_constraints_added=False)
    report['passed']=bool(jump.failure is None and jump.phase=='RECOVER'
                           and data.time>=args.duration-.01
                           and report['flight_time_s'] is not None and report['flight_time_s']>.05
                           and report['max_two_wheel_clearance_m']>.008
                           and report['takeoff_com_velocity_m_s']>0
                           and ballistic_acceleration is not None
                           and abs(ballistic_acceleration+9.81)<.5
                           and report['tail_pitch_rms_deg']<1 and report['tail_roll_rms_deg']<1
                           and report['tail_yaw_rms_deg']<1
                           and report['tail_length_error_m']<.005
                           and abs(report['final_position_error_m'])<.05
                           and post_landing_max_position_error is not None
                           and post_landing_max_position_error<.05
                           and post_landing_max_speed<.35
                           and (jump.tuck_length is None or tuck_achieved)
                           and abs(report['final_forward_speed_m_s'])<.05)
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (output/'parameters.json').write_text(json.dumps(dict(crouch=args.crouch,
        extension=args.extension,thrust_time=args.thrust_time,boost=args.boost,
        phases=jump.PHASES,motor_profile=args.motor_profile,cut_length=jump.cut_length,
        takeoff_momentum_gain=jump.takeoff_momentum_gain,
        takeoff_momentum_target_Nms=jump.takeoff_momentum_target,
        tuck_length=jump.tuck_length,air_actions=jump.AIR_ACTIONS,
        force_limit_N=args.force_limit,deep_schedule=args.deep_schedule,wheel_contact_time=args.wheel_contact_time,
        controller_version=CONTROLLER_VERSION),indent=2),encoding='utf-8')
    if not args.no_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(6,1,figsize=(11,14),sharex=True)
        axes[0].plot(a[:,0],a[:,3],label='COM z');axes[0].plot(a[:,0],a[:,2],label='chassis z')
        axes[0].set_ylabel('Height (m)')
        for col,label in [(5,'left'),(6,'right')]:axes[1].plot(a[:,0],a[:,col],label=label)
        axes[1].set_ylabel('Wheel clearance (m)')
        for col,label in [(9,'left'),(10,'right'),(11,'target')]:axes[2].plot(a[:,0],a[:,col],label=label)
        axes[2].set_ylabel('Leg length (m)')
        for col,label in [(12,'roll'),(13,'pitch')]:axes[3].plot(a[:,0],np.rad2deg(a[:,col]),label=label)
        axes[3].set_ylabel('Attitude (deg)')
        axes[4].plot(a[:,0],(a[:,15]-jump.reference_x)*100,label='axle position error')
        axes[4].set_ylabel('Forward displacement (cm)')
        axes[5].step(a[:,0],a[:,1],where='post',label='phase')
        axes[5].set_yticks(range(len(jump.PHASES)),jump.PHASES);axes[5].set_xlabel('Time (s)')
        for ax in axes:ax.grid(alpha=.3);ax.legend(loc='upper right')
        fig.suptitle('Motor-driven jump on free-floating robot.xml')
        fig.tight_layout();fig.savefig(output/'response.png',dpi=150);plt.close(fig)
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--viewer',action='store_true')
    p.add_argument('--duration',type=float,default=10.)
    p.add_argument('--crouch',type=float,default=.14)
    p.add_argument('--extension',type=float,default=.225)
    p.add_argument('--thrust-time',type=float,default=.10)
    p.add_argument('--boost',type=float,default=20.)
    p.add_argument('--force-limit',type=float,default=80.,help='Virtual radial force bound per leg; actual motor torque/speed limits remain active')
    p.add_argument('--deep-schedule',action='store_true',help='Also design grounded LQR gains at 8, 10 and 12 cm')
    p.add_argument('--wheel-contact-time',type=float,default=None,help='Explicit wheel contact solref time in seconds; omit for XML default')
    p.add_argument('--output',default='results/closed_loop/jump')
    p.add_argument('--no-plot',action='store_true')
    p.add_argument('--motor-profile',choices=['xml','dm4310-v11-24v','dm4310-v11-peak-only'],default='dm4310-v11-24v')
    p.add_argument('--cut-length',type=float,default=None,help='End thrust at measured leg length; default preserves original demo')
    p.add_argument('--takeoff-momentum-gain',type=float,default=None,help='Centroidal pitch momentum feedback; use zero with --no-tuck for the old baseline')
    p.add_argument('--takeoff-momentum-target',type=float,default=None,help='Desired takeoff pitch angular momentum, N m s')
    p.add_argument('--tuck-length',type=float,default=.13,help='Airborne hip-to-wheel length, m')
    p.add_argument('--no-tuck',action='store_true',help='Use the previous shallow retraction timing')
    args=p.parse_args()
    if args.duration<=0 or args.thrust_time<=0:p.error('Duration and thrust time must be positive')
    if not (.08<=args.crouch<.18<args.extension<=.23):
        p.error('Require 0.08 <= crouch < 0.18 < extension <= 0.23 m')
    if not 0<args.force_limit<=400:p.error('Force limit must be in (0,400] N per leg')
    if args.wheel_contact_time is not None and args.wheel_contact_time<.001:p.error('Wheel contact time must be >=.001 s')
    if args.cut_length is not None and not args.crouch<args.cut_length<=.23:
        p.error('Cut length must be between crouch length and 0.23 m')
    if not .08<=args.tuck_length<=.17:p.error('Tuck length must be between .08 and .17 m')
    run(args)
