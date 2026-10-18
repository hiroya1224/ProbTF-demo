"""Fixed native observation charts evaluated on finite candidate geometry."""
from dataclasses import dataclass
import math

import numpy as np
from scipy.special import logsumexp

from probtf.geometry import (quat_to_rotmat, rotmat_to_quat,
    quaternion_to_rotation_vector, skew, tangent_basis)
from .finite_modal import finite_modal_path
from .sensors import prepare_marker_observation, _marker_mode_chart, _rotation_log_right_jacobian_inverse


def _readonly(value):
    result=np.array(value,dtype=float,copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError('Native nonlinear observation parameters must be finite.')
    result.setflags(write=False)
    return result


class SphereLogCut(ValueError):
    """A finite antipodal direction has no unique sphere-log derivative."""


def sphere_log_residual(direction, observed, basis):
    """Return Log_observed(direction) in a fixed 2D plane and its derivative.

The homogeneous extension makes radial perturbations vanish; finite geometry
supplies unit directions. Only the exact antipodal cut is undefined. The
small-angle series evaluates the same function, without truncating angles.
"""
    direction=np.asarray(direction,dtype=float)
    observed=np.asarray(observed,dtype=float)
    basis=np.asarray(basis,dtype=float)
    if (direction.shape!=(3,) or observed.shape!=(3,) or basis.shape!=(3,2)
            or not np.all(np.isfinite(direction))):
        raise ValueError('Sphere-log coordinates require finite three-vectors and a fixed 3x2 basis.')
    v=basis.T@direction
    s=float(np.linalg.norm(v));c=float(observed@direction)
    norm=math.hypot(s,c)
    if norm==0.:
        raise ValueError('A zero direction has no sphere-log observation coordinate.')
    # An exactly antiparallel machine vector is a cut even when rounded
    # tangent-basis orthogonality makes B.T @ direction slightly nonzero.
    # This is an exact domain check, never a near-antipode angle threshold.
    if c<0. and (s==0. or np.all(np.cross(direction,observed)==0.)):
        raise SphereLogCut('Finite antipodal IMU direction: scalar angular likelihood is finite but the sphere-log axis/derivative is not unique.')
    theta=math.atan2(s,c)
    if theta<1e-4:
        square=theta*theta
        f=(1.+square/6.+7.*square*square/360.+31.*square**3/15120.)/norm
        alpha=-(2./3.+square/5.+17.*square*square/420.)/norm**3
    else:
        f=theta/s
        alpha=(s*c/norm**2-theta)/s**3
    residual=f*v
    jacobian=f*basis.T+np.outer(v,alpha*(v@basis.T)-observed/norm**2)
    return residual,jacobian


@dataclass(frozen=True)
class FixedLikelihood:
    label: str
    sensor: str
    sensor_id: str
    observation_id: str
    path: tuple
    covariance: np.ndarray
    log_weight: float=0.
    translation: object=None
    rotation: object=None
    root_direction: object=None
    observed_direction: object=None
    basis: object=None

    def __post_init__(self):
        for name in ('covariance','translation','rotation','root_direction','observed_direction','basis'):
            if getattr(self,name) is not None:
                object.__setattr__(self,name,_readonly(getattr(self,name)))
        object.__setattr__(self,'path',tuple(self.path))
        if math.isnan(self.log_weight) or self.log_weight==math.inf:
            raise ValueError('A native observation log weight must be finite or negative infinity.')

    def evaluate(self,state,delta):
        pose=finite_modal_path(state,self.path,delta)
        rotation=quat_to_rotmat(pose.rotation_wxyz)
        if self.sensor=='marker':
            vector=quaternion_to_rotation_vector(rotmat_to_quat(self.rotation.T@rotation))
            residual=np.concatenate((pose.translation-self.translation,vector))
            jacobian=np.vstack((pose.J_t,_rotation_log_right_jacobian_inverse(vector)@pose.J_right_rot))
            return residual,jacobian
        direction=rotation@self.root_direction
        residual,derivative=sphere_log_residual(direction,self.observed_direction,self.basis)
        J_direction=-rotation@skew(self.root_direction)@pose.J_right_rot
        return residual,derivative@J_direction

    def record(self):
        return dict(label=self.label,sensor=self.sensor,sensor_id=self.sensor_id,
            observation_id=self.observation_id,path=list(self.path),covariance=self.covariance.tolist(),
            log_weight=self.log_weight,
            fixed_native_chart=True,prior_dependent_chart=False,
            point_dependent_chart_volume_applied=False,
            translation=None if self.translation is None else self.translation.tolist(),
            rotation=None if self.rotation is None else self.rotation.tolist(),
            observed_direction=None if self.observed_direction is None else self.observed_direction.tolist(),
            root_direction=None if self.root_direction is None else self.root_direction.tolist(),
            basis=None if self.basis is None else self.basis.tolist())


@dataclass(frozen=True)
class PreparedLikelihood:
    observation: dict
    components: tuple
    diagnostics: dict
    prepared_marker: object=None


def prepare_likelihood(common,observation,paths,image,stamp):
    """Prepare image-only native branches or one fixed gravity likelihood."""
    sensor=observation['sensor']
    if sensor=='imu':
        root=np.array(observation['gravity_in_root'],dtype=float,copy=True)
        observed=np.array(observation['gravity_in_sensor'],dtype=float,copy=True)
        if np.linalg.norm(root)==0. or np.linalg.norm(observed)==0.:
            raise ValueError('Gravity directions must be nonzero.')
        root/=np.linalg.norm(root);observed/=np.linalg.norm(observed)
        basis=tangent_basis(observed)
        target=paths['links'][observation['target_link']]
        inverse=tuple((edge,not flag)for edge,flag in reversed(target))
        sigma=float(observation['direction_sigma_rad'])
        if not math.isfinite(sigma) or sigma<=0.:
            raise ValueError('The angular IMU likelihood needs a positive finite standard deviation.')
        likelihood=FixedLikelihood('gravity','imu',observation['sensor_id'],observation['observation_id'],
            inverse,np.eye(2)*sigma**2,root_direction=root,observed_direction=observed,basis=basis)
        return PreparedLikelihood(dict(observation),(likelihood,),dict(sensor='imu',
            target_link=observation['target_link'],coordinates='observed-fixed S2 Log tangent plane (2D)',
            shared_observation_frame=True,tangent_basis=basis.tolist(),observed_direction=observed.tolist(),
            information_rank=2,heading_observation=False,measurement_variance=sigma**2,
            approximation='finite angular extension of the fixed two-dimensional Gaussian tangent likelihood; scalar likelihood with no point-dependent volume'))
    if sensor!='marker':
        raise ValueError('Unknown nonlinear observation sensor: '+str(sensor))
    prepared=prepare_marker_observation(common,image,stamp)
    diagnostics=dict(sensor='marker',detected=prepared.detected is not None,
        marker_board_fusion=prepared.board_fusion,
        image_approximation='all native original image-Hessian joint Laplace Gaussians in their own fixed modal charts',
        detector_prior='translation prior explicitly disabled; image-only local likelihood mixture',
        fixed_native_chart=True,prior_dependent_chart=False,
        point_dependent_chart_volume_applied=False)
    if prepared.detected is None:
        return PreparedLikelihood(dict(observation),(),diagnostics,prepared)
    detected=prepared.detected
    logs=np.asarray(detected.log_masses)-logsumexp(detected.log_masses)
    components=[]
    for native,log_weight in zip(detected.record.distribution.components,logs):
        rotation=quat_to_rotmat(native.orientation.mode_wxyz)
        gaussian,volume=_marker_mode_chart(native,rotation)
        # The chart is the component's original mode, so its Jacobian is I.
        # The scalar likelihood is evaluated directly in that fixed chart;
        # no parent-specific volume is subsequently applied.
        components.append(FixedLikelihood(native.component_id,'marker',observation['sensor_id'],
            observation['observation_id'],paths['camera_marker'],gaussian.covariance,float(log_weight),
            gaussian.mean[:3],rotation))
    diagnostics['components']=[component.record()for component in components]
    diagnostics['native_log_weights']=[float(value)for value in logs]
    return PreparedLikelihood(dict(observation),tuple(components),diagnostics,prepared)
