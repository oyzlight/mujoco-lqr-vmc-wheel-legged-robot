"""Explicit, approximate actuator envelope; not a measured motor model."""
import numpy as np


PROFILE = 'dm4310-v11-24v'
PEAK_ONLY = 'dm4310-v11-peak-only'


def configure(model, name):
    if name == 'xml':
        return
    if name not in [PROFILE,PEAK_ONLY]:
        raise ValueError(name)
    # Manufacturer output-shaft torques: the J4310 internal 10:1 gearbox is
    # ALREADY included. No extra factor of ten is applied.
    for a in range(model.nu):
        peak = 2. if a in [2,5] else 7.
        model.actuator_ctrlrange[a] = [-peak,peak]


def limit(model, data, command, name):
    if name in ['xml',PEAK_ONLY]:
        return command.copy()
    out=command.copy()
    for a in range(model.nu):
        joint=model.actuator_trnid[a,0]
        speed=data.qvel[model.jnt_dofadr[joint]]
        peak=2. if a in [2,5] else 7.
        rpm=320. if a in [2,5] else 200.
        omega0=rpm*2*np.pi/60
        # Straight-line motoring envelope anchored at peak torque and no-load
        # speed. This is an ENGINEERING APPROXIMATION, not the vendor curve.
        available=peak*max(0.,1-abs(speed)/omega0) if command[a]*speed>0 else peak
        out[a]=np.clip(command[a],-available,available)
    return out
