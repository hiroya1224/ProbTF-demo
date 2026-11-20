"""Memory-bounded complete-prefix likelihood evaluation of unchanged RK4."""
from pathlib import Path
import ctypes
import hashlib
import importlib.util
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_full_rate_existing_native',HERE/'native_flow.py')
_native=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_native;_spec.loader.exec_module(_native)
_LIBRARY=None


def _library():
    global _LIBRARY
    if _LIBRARY is not None:return _LIBRARY
    source=HERE/'full_rate_flow.cpp'
    digest=hashlib.sha256(source.read_bytes()+(HERE/'native_flow.cpp').read_bytes()).hexdigest()[:20]
    directory=Path('/tmp')/('probtf-full-rate-native-'+digest)
    directory.mkdir(parents=True,exist_ok=True)
    target=directory/'full_rate_flow.so'
    if not target.exists():
        descriptor,name=tempfile.mkstemp(prefix='full_rate_',suffix='.so',dir=directory)
        os.close(descriptor);candidate=Path(name)
        try:
            subprocess.run(['g++','-O3','-std=c++17','-fPIC','-shared',str(source),'-o',str(candidate)],
                check=True,capture_output=True,text=True)
            os.replace(candidate,target)
        finally:
            if candidate.exists():candidate.unlink()
    library=ctypes.CDLL(str(target));pointer=ctypes.POINTER(ctypes.c_double)
    library.probtf_full_rate_prefix.argtypes=[ctypes.c_int,pointer,pointer,ctypes.c_int,pointer,pointer,
        ctypes.c_int,ctypes.c_int,pointer,pointer,pointer,pointer,pointer,ctypes.c_double,
        ctypes.c_double,ctypes.c_double,ctypes.c_int,pointer,pointer,pointer,ctypes.POINTER(ctypes.c_longlong)]
    library.probtf_full_rate_prefix.restype=ctypes.c_int
    _LIBRARY=library
    return library


class FullRateFlow(_native.ControlledFlow):
    def prefix_and_block(self,nodes,times,observations,prefix,finish,covariance,student_df=None):
        """Return past j=1..prefix-1 log likelihood and states prefix..finish.

        Future states in a block depend only on known deterministic input;
        no future observation enters these prefix likelihoods.
        ``covariance`` is the observation's marginal covariance in both
        modes. For Student-t with df>2, its scale is (df-2)/df times that
        covariance. None retains the original Gaussian computation.
        """
        nodes=np.ascontiguousarray(nodes,dtype=float)
        stamps=np.ascontiguousarray(times[:finish+1],dtype=float)
        # The native physics preview receives no future/current reading:
        # only j<prefix can contribute to its accumulated past likelihood.
        measured=np.ascontiguousarray(observations[:prefix],dtype=float)
        covariance=np.asarray(covariance,float)
        if (nodes.ndim!=2 or nodes.shape[1]!=3 or not len(nodes) or not np.isfinite(nodes).all()
                or stamps.ndim!=1 or len(stamps)!=finish+1 or not np.isfinite(stamps).all()
                or stamps[0]!=self.start or stamps[-1]>self.end or np.any(np.diff(stamps)<=0.)
                or measured.shape!=(prefix,3) or not np.isfinite(measured).all()
                or not 0<=prefix<=finish or covariance.shape!=(3,3)):
            raise ValueError('Invalid finite prefix/block input')
        if student_df is None:
            df=0.
            root=np.linalg.cholesky(covariance)
            precision=np.ascontiguousarray(np.linalg.solve(covariance,np.eye(3)))
            constant=-1.5*np.log(2*np.pi)-np.log(np.diag(root)).sum()
        else:
            df=float(student_df)
            if not np.isfinite(df) or df<=2.:
                raise ValueError('Student-t df must exceed 2 for a declared finite marginal covariance')
            scale=((df-2.)/df)*covariance
            root=np.linalg.cholesky(scale)
            precision=np.ascontiguousarray(np.linalg.solve(scale,np.eye(3)))
            constant=(math.lgamma((df+3.)/2.)-math.lgamma(df/2.)
                      -1.5*np.log(df*np.pi)-np.log(np.diag(root)).sum())
        log_likelihood=np.empty(len(nodes));states=np.empty((finish-prefix+1,len(nodes),7))
        norm=ctypes.c_double();steps=ctypes.c_longlong();pointer=ctypes.POINTER(ctypes.c_double)
        ptr=lambda array:array.ctypes.data_as(pointer)
        status=_library().probtf_full_rate_prefix(len(nodes),ptr(nodes),ptr(self.q0),len(stamps),ptr(stamps),
            ptr(measured),prefix,len(self.torque_time),ptr(self.torque_time),ptr(self.torque_value),
            ptr(self.tensor),ptr(self.inverse),ptr(precision),constant,df,self.max_step,self.nonlinear,
            ptr(log_likelihood),ptr(states),ctypes.byref(norm),ctypes.byref(steps))
        if status==1:raise ValueError('Native prefix flow received invalid integration arguments')
        if status:raise FloatingPointError('Native prefix flow produced a non-finite state or undefined quaternion')
        self.total_steps+=steps.value;self.maximum_norm_correction=max(self.maximum_norm_correction,norm.value)
        return log_likelihood,states
