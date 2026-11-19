"""ctypes batch for the unchanged controlled Euler/quaternion RK4 equations.

The compiled binary is a reproducible temporary cache, never a repository
artifact.  The wrapper only validates inputs required to evaluate the flow.
It does not accept/reject observations or alter any statistical covariance.
"""
from pathlib import Path
import ctypes
import hashlib
import os
import subprocess
import tempfile

import numpy as np

HERE=Path(__file__).resolve().parent
_LIBRARY=None


def _library():
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    source=HERE/'native_flow.cpp'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    directory=Path('/tmp')/('probtf-synthetic-native-'+digest)
    directory.mkdir(parents=True,exist_ok=True)
    target=directory/'controlled_flow.so'
    if not target.exists():
        fd,name=tempfile.mkstemp(prefix='controlled_flow_',suffix='.so',dir=directory)
        os.close(fd)
        candidate=Path(name)
        try:
            subprocess.run(['g++','-O3','-std=c++17','-fPIC','-shared',str(source),
                            '-o',str(candidate)],check=True,capture_output=True,text=True)
            os.replace(candidate,target)
        finally:
            if candidate.exists(): candidate.unlink()
    library=ctypes.CDLL(str(target))
    pointer=ctypes.POINTER(ctypes.c_double)
    library.probtf_controlled_flow.argtypes=[ctypes.c_int,pointer,pointer,ctypes.c_int,pointer,
        ctypes.c_int,pointer,pointer,pointer,pointer,ctypes.c_double,ctypes.c_int,pointer,
        pointer,ctypes.POINTER(ctypes.c_longlong)]
    library.probtf_controlled_flow.restype=ctypes.c_int
    _LIBRARY=library
    return library


def torque_arrays(profile,start,end):
    """Use the actual stored piecewise-linear profile, without changing knots."""
    torque=profile['torque']
    if hasattr(torque,'time') and hasattr(torque,'torque'):
        time=np.asarray(torque.time,dtype=float)
        values=np.asarray(torque.torque,dtype=float)
        residual=np.asarray(getattr(torque,'residual_mean',np.zeros(3)),float)
        values=values+residual
    elif callable(torque):
        # A generic callable has no declared knots. This fallback explicitly
        # represents it on the experiment's recorded-control 1 kHz clock.
        time=np.linspace(start,end,max(1,int(np.ceil((end-start)/.001)))+1)
        values=np.stack([torque(float(t)) for t in time])
    else:
        time=np.array([start,end],float)
        values=np.broadcast_to(np.asarray(torque,float),(2,3)).copy()
    if (time.ndim!=1 or len(time)<2 or values.shape!=(len(time),3)
            or not np.isfinite(time).all() or not np.isfinite(values).all()
            or np.any(np.diff(time)<=0.) or time[0]>start or time[-1]<end):
        raise ValueError('Known torque must cover the experiment with finite increasing knots')
    return np.ascontiguousarray(time),np.ascontiguousarray(values)


class ControlledFlow:
    def __init__(self,profile,q0,start,end,max_step=.002,mode='nonlinear'):
        self.tensor=np.ascontiguousarray(profile['inertia'],dtype=float)
        if self.tensor.shape!=(3,3) or not np.isfinite(self.tensor).all():
            raise ValueError('Finite 3x3 inertia required')
        self.tensor=np.ascontiguousarray(.5*(self.tensor+self.tensor.T))
        np.linalg.cholesky(self.tensor)
        self.inverse=np.ascontiguousarray(np.linalg.solve(self.tensor,np.eye(3)))
        self.q0=np.ascontiguousarray(q0,dtype=float)
        if self.q0.shape!=(4,) or not np.isfinite(self.q0).all() or np.linalg.norm(self.q0)==0.:
            raise ValueError('Finite nonzero initial quaternion required')
        self.q0=self.q0/np.linalg.norm(self.q0)
        self.start,self.end,self.max_step=float(start),float(end),float(max_step)
        if not np.isfinite(self.max_step) or self.max_step<=0.:
            raise ValueError('Positive finite integration step required')
        if mode not in ('nonlinear','linear_forced'):
            raise ValueError('Unknown angular dynamics mode')
        self.nonlinear=int(mode=='nonlinear')
        self.torque_time,self.torque_value=torque_arrays(profile,self.start,self.end)
        self.total_steps=0
        self.maximum_norm_correction=0.

    def evaluate(self,initial_omega,times):
        nodes=np.ascontiguousarray(initial_omega,dtype=float)
        stamps=np.ascontiguousarray(times,dtype=float)
        if (nodes.ndim!=2 or nodes.shape[1]!=3 or not len(nodes) or not np.isfinite(nodes).all()
                or stamps.ndim!=1 or not len(stamps) or not np.isfinite(stamps).all()
                or stamps[0]!=self.start or stamps[-1]>self.end or np.any(np.diff(stamps)<=0.)):
            raise ValueError('Finite nodes and increasing output times starting at initial time required')
        output=np.empty((len(stamps),len(nodes),7),dtype=float)
        norm=ctypes.c_double()
        steps=ctypes.c_longlong()
        pointer=ctypes.POINTER(ctypes.c_double)
        ptr=lambda value:value.ctypes.data_as(pointer)
        status=_library().probtf_controlled_flow(len(nodes),ptr(nodes),ptr(self.q0),len(stamps),ptr(stamps),
            len(self.torque_time),ptr(self.torque_time),ptr(self.torque_value),ptr(self.tensor),
            ptr(self.inverse),self.max_step,self.nonlinear,ptr(output),ctypes.byref(norm),ctypes.byref(steps))
        if status==1: raise ValueError('Native controlled flow received invalid integration arguments')
        if status: raise FloatingPointError('Native controlled RK4 produced a non-finite state or zero quaternion')
        self.total_steps+=steps.value
        self.maximum_norm_correction=max(self.maximum_norm_correction,norm.value)
        return output
