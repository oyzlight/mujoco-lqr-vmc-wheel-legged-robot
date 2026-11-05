"""Closed topology, CAD alignment, differential kinematics and motor wiring checks."""
import json
from pathlib import Path
import sys
import mujoco
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lqr_vmc import LqrVmc


def main():
    reports={}
    for filename in ['robot.xml','robot_planar.xml','robot_fixed.xml']:
        m=mujoco.MjModel.from_xml_path(str(ROOT/filename));d=mujoco.MjData(m)
        mujoco.mj_forward(m,d)
        assert m.nu==6 and m.neq==8
        assert np.all(m.body_mass[1:]>0) and np.all(m.body_inertia[1:]>0)
        assert [m.joint(int(j)).name for j in m.actuator_trnid[:,0]]==[s+'_'+n for s in ['left','right'] for n in ['hip_A','hip_B','wheel_joint']]
        rows=d.efc_type[:d.nefc]==mujoco.mjtConstraint.mjCNSTR_EQUALITY
        rank=int(np.linalg.matrix_rank(d.efc_J.reshape(d.nefc,m.nv)[rows],tol=1e-8))
        assert rank==12
        maximum=0.
        dtype=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')])
        for item in json.loads((ROOT/'closed_meshes/manifest.json').read_text())['groups']:
            g=m.geom(item['group']+'_visual').id;mid=m.geom_dataid[g]
            start,count=m.mesh_vertadr[mid],m.mesh_vertnum[mid]
            actual=m.mesh_vert[start:start+count]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            source=np.frombuffer((ROOT/'closed_meshes'/item['file']).read_bytes()[84:],dtype=dtype)['v'].reshape(-1,3).astype(float)
            source[:,2]+=m.body_pos[m.body('chassis').id,2]
            maximum=max(maximum,float(np.max(abs(source.min(0)-actual.min(0)))),float(np.max(abs(source.max(0)-actual.max(0)))))
        assert maximum<2e-6,maximum
        reports[filename]=dict(nq=m.nq,nv=m.nv,nu=m.nu,neq=m.neq,equality_rank=rank,physical_dofs=m.nv-rank,visual_bounds_error_m=maximum,mass_kg=float(m.body_mass.sum()))
    c=LqrVmc();leg=c.leg;qstand=leg.ik(.18,0)
    max_jac=max_pose=max_work=0.
    for q in [np.zeros(2),qstand,qstand+[.04,-.02],qstand+[-.03,.05]]:
        f=leg.forward(q,True);h=1e-6
        jfd=np.column_stack([(leg.forward(q+np.eye(2)[i]*h)['W']-leg.forward(q-np.eye(2)[i]*h)['W'])/(2*h) for i in range(2)])
        max_jac=max(max_jac,float(np.max(abs(jfd-f['Jcart']))))
        d=mujoco.MjData(c.model)
        for side in c.sides:c.set_leg_pose(d,side,q)
        mujoco.mj_forward(c.model,d)
        for side in c.sides:
            actual=(d.xpos[side['wheel']]-d.xpos[side['thigh']])[[0,2]]
            max_pose=max(max_pose,float(np.linalg.norm(actual-f['W'])))
        closure,gear=c.constraint_errors(d)
        assert closure<1e-10 and gear<1e-10
        _,_,j=leg.polar(q);rate=np.array([.2,-.3]);wrench=np.array([20.,.4])
        max_work=max(max_work,float(abs((-j.T@wrench)@rate-wrench@(j@(-rate)))))
    assert max_jac<1e-6 and max_pose<1e-9 and max_work<1e-12
    reports['kinematics']=dict(jacobian_FD_error=max_jac,MuJoCo_FK_error_m=max_pose,virtual_work_error_W=max_work)
    # The chassis is fixed and gravity disabled ONLY for independent actuator tests.
    m=mujoco.MjModel.from_xml_path(str(ROOT/'robot_fixed.xml'));m.opt.gravity[:]=0
    responses=[]
    for actuator in range(m.nu):
        d=mujoco.MjData(m)
        for side in ['left','right']:
            names=['hip_A','hip_B','output_A','output_B','rocker_pin','rod29_pin','lower_pin','rod35_pin','wheel_joint']
            for n,value in zip(names,c.leg_coordinates(qstand)):
                d.qpos[m.jnt_qposadr[m.joint(side+'_'+n).id]]=value
        initial=d.qpos.copy();d.ctrl[actuator]=.2
        for _ in range(100):mujoco.mj_step(m,d)
        mujoco.mj_forward(m,d)
        jid=int(m.actuator_trnid[actuator,0]);motion=float(d.qpos[m.jnt_qposadr[jid]]-initial[m.jnt_qposadr[jid]])
        assert np.isfinite(d.qpos).all() and abs(motion)>1e-5
        loop=max(np.linalg.norm(d.site_xpos[m.site(s+'_'+p+'0').id]-d.site_xpos[m.site(s+'_'+p+'1').id]) for s in ['left','right'] for p in ['C','D'])
        gear=max(abs(d.qpos[m.jnt_qposadr[m.joint(s+'_hip_'+a).id]]+d.qpos[m.jnt_qposadr[m.joint(s+'_output_'+a).id]]) for s in ['left','right'] for a in ['A','B'])
        assert loop<.0002 and gear<.005,(loop,gear)
        responses.append(dict(actuator=m.actuator(actuator).name,angle_change_rad=motion,loop_error_m=float(loop),gear_error_rad=float(gear)))
    reports['six_motor_pulses']=responses
    d=c.solve_equilibrium();c.linearize();c.initialize(d)
    reports['equilibrium']=c.equilibrium_info
    reports['lqr_max_pole']=c.design_info['max_pole_magnitude']
    reports['fixed_point_drift']=c.design_info['linearization_fixed_point_drift']
    camera=mujoco.MjvCamera();camera.lookat[:]=d.xpos[c.chassis]
    camera.distance=.85;camera.azimuth=125;camera.elevation=-18
    option=mujoco.MjvOption();option.geomgroup[3]=0;option.sitegroup[4]=0
    with mujoco.Renderer(c.model,height=1000,width=1440) as renderer:
        renderer.update_scene(d,camera=camera,scene_option=option)
        Image.fromarray(renderer.render()).save(ROOT/'preview.png')
    reports['passed']=True
    (ROOT/'validation.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
