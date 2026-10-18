"""Fixed native observations and one pass of core moment propagation.

Marker chart transport is the existing local first-order coordinate change;
its input mean retains the core O(P) correction. It does not define a new
Gaussian law globally on rotations or add a candidate-dependent volume.
"""
from dataclasses import dataclass,replace
import math
import numpy as np
from scipy.special import logsumexp

from probtf.geometry import (quat_to_rotmat,quaternion_to_rotation_vector,
                            rotmat_to_quat,tangent_basis)
from probtf.spherical_law.first_order import linear_query
from .body_native import pose_query,direction_query
from .sensors import (prepare_marker_observation,_marker_mode_chart,
                      _rotation_log_right_jacobian_inverse)


def _readonly(value):
    out=np.array(value,dtype=float,copy=True)
    if not np.all(np.isfinite(out)):
        raise ValueError('A native likelihood parameter must be finite.')
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class NativeMomentLikelihood:
    label: str
    sensor: str
    sensor_id: str
    observation_id: str
    path: tuple
    observed_mean: np.ndarray
    covariance: np.ndarray
    log_weight: float=0.
    rotation_wxyz: object=None
    root_direction: object=None
    observed_direction: object=None
    basis: object=None

    def __post_init__(self):
        for key in ('observed_mean','covariance','rotation_wxyz','root_direction','observed_direction','basis'):
            value=getattr(self,key)
            if value is not None:object.__setattr__(self,key,_readonly(value))
        object.__setattr__(self,'path',tuple(self.path))
        if math.isnan(self.log_weight)or self.log_weight==math.inf:
            raise ValueError('A native mixture log weight must be finite or negative infinity.')

    def record(self):
        def array(value):return None if value is None else value.tolist()
        return dict(label=self.label,sensor=self.sensor,sensor_id=self.sensor_id,
            observation_id=self.observation_id,path=list(self.path),
            observed_mean=self.observed_mean.tolist(),covariance=self.covariance.tolist(),
            log_weight=self.log_weight,rotation_wxyz=array(self.rotation_wxyz),
            root_direction=array(self.root_direction),observed_direction=array(self.observed_direction),
            basis=array(self.basis),fixed_native_chart=True,prior_dependent_native_likelihood=False,
            point_dependent_chart_volume_applied=False)


@dataclass(frozen=True)
class PropagatedMomentObservation:
    likelihood: NativeMomentLikelihood
    query: object
    source_query: object
    chart_matrix: np.ndarray
    chart_offset: np.ndarray

    def record(self):
        q=self.query
        return dict(native_component_label=self.likelihood.label,
            observation_id=self.likelihood.observation_id,sensor_id=self.likelihood.sensor_id,
            source_state_id=q.state_id,source_component_id=q.component_id,
            source_stamp=q.stamp,source_command=list(q.command),source_version=q.version,
            source_query_id=self.source_query.query_id,query_id=q.query_id,
            propagated_mean=q.mean.tolist(),propagated_covariance=q.covariance.tolist(),
            full_body_observation_cross_covariance=q.cross_covariance.tolist(),
            jacobian=q.jacobian.tolist(),within=q.within.tolist(),between=q.between.tolist(),
            mode=q.mode.tolist(),correction=q.correction.tolist(),
            physical_modal_rotation_wxyz=(None if self.source_query.rotation_wxyz is None
                else self.source_query.rotation_wxyz.tolist()),
            fixed_observation_chart_rotation_wxyz=(None if self.likelihood.rotation_wxyz is None
                else self.likelihood.rotation_wxyz.tolist()),
            chart_matrix=self.chart_matrix.tolist(),chart_offset=self.chart_offset.tolist(),
            core_O_P_mean_used=True,core_within_between_retained=True,
            chart_transport='one fixed local first-order coordinate transport; not iterated statistical linearization',
            additional_chart_Hessian_mean_correction_added=False)


@dataclass(frozen=True)
class PreparedMomentObservation:
    observation: dict
    components: tuple
    diagnostics: dict
    prepared_marker: object=None

    def propagate(self,state):
        if not self.components:return ()
        native=self.components[0]
        if native.sensor=='marker':
            source=pose_query(state,native.path,query_id=native.sensor_id+':camera_marker:core_moments',frame='camera')
        else:
            source=direction_query(state,native.path,native.root_direction,
                query_id=native.sensor_id+':gravity:core_moments',frame=self.observation['target_link'])
        result=[]
        for native in self.components:
            if native.sensor=='marker':
                phi=quaternion_to_rotation_vector(rotmat_to_quat(
                    quat_to_rotmat(native.rotation_wxyz).T@quat_to_rotmat(source.rotation_wxyz)))
                M=np.eye(6);M[3:,3:]=_rotation_log_right_jacobian_inverse(phi)
                offset=np.r_[np.zeros(3),phi]
                query=linear_query(source,M,query_id=native.sensor_id+':fixed_native:'+native.label)
                # Keep PointQuery.rotation_wxyz's physical modal rotation.
                # The fixed observation chart reference is separate metadata
                # in the native likelihood and this observation's record.
                query=replace(query,mean=query.mean+offset,mode=query.mode+offset)
            else:
                M=native.basis.T;offset=np.zeros(2)
                query=linear_query(source,M,query_id=native.sensor_id+':observed_fixed_tangent2')
            result.append(PropagatedMomentObservation(native,query,source,_readonly(M),_readonly(offset)))
        return tuple(result)


def prepare_observation(common,observation,paths,image,stamp):
    """Prepare one image once, without a body/camera/Keep/Reset prior."""
    sensor=observation['sensor']
    if sensor=='imu':
        root=np.array(observation['gravity_in_root'],dtype=float,copy=True)
        observed=np.array(observation['gravity_in_sensor'],dtype=float,copy=True)
        if (root.shape!=(3,)or observed.shape!=(3,)or not np.all(np.isfinite(root))
                or not np.all(np.isfinite(observed))or np.linalg.norm(root)==0.or np.linalg.norm(observed)==0.):
            raise ValueError('IMU moment propagation needs finite nonzero three-dimensional directions.')
        root/=np.linalg.norm(root);observed/=np.linalg.norm(observed)
        basis=tangent_basis(observed);sigma=float(observation['direction_sigma_rad'])
        if not math.isfinite(sigma)or sigma<=0.:
            raise ValueError('The IMU likelihood needs a positive finite angular standard deviation.')
        target=paths['links'][observation['target_link']]
        inverse=tuple((edge,not flag)for edge,flag in reversed(target))
        native=NativeMomentLikelihood('gravity','imu',observation['sensor_id'],observation['observation_id'],
            inverse,basis.T@observed,np.eye(2)*sigma**2,
            root_direction=root,observed_direction=observed,basis=basis)
        return PreparedMomentObservation(dict(observation),(native,),dict(sensor='imu',
            target_link=observation['target_link'],coordinates='observed-fixed S2 tangent plane (2D) of propagated core direction moments',
            shared_observation_frame=True,tangent_basis=basis.tolist(),observed_direction=observed.tolist(),
            information_rank=2,heading_observation=False,measurement_variance=sigma**2,
            approximation='one local Gaussian tangent observation; finite antipodal direction ambiguity is not resolved by this projection',
            point_S2_Log_likelihood_used=False))
    if sensor!='marker':raise ValueError('Unknown moment observation sensor: '+str(sensor))
    prepared=prepare_marker_observation(common,image,stamp)
    diagnostics=dict(sensor='marker',detected=prepared.detected is not None,
        marker_board_fusion=prepared.board_fusion,
        image_approximation='every original image-Hessian joint Laplace Gaussian in its own fixed native chart',
        detector_prior='translation prior explicitly disabled; image-only local likelihood mixture',
        fixed_native_chart=True,prior_dependent_chart=False,point_dependent_chart_volume_applied=False)
    if prepared.detected is None:return PreparedMomentObservation(dict(observation),(),diagnostics,prepared)
    detected=prepared.detected
    logs=np.asarray(detected.log_masses)-logsumexp(detected.log_masses)
    components=[]
    for component,log_weight in zip(detected.record.distribution.components,logs):
        quaternion=component.orientation.mode_wxyz
        law,_unused_volume=_marker_mode_chart(component,quat_to_rotmat(quaternion))
        components.append(NativeMomentLikelihood(component.component_id,'marker',observation['sensor_id'],
            observation['observation_id'],paths['camera_marker'],law.mean,law.covariance,
            float(log_weight),rotation_wxyz=quaternion))
    diagnostics['components']=[item.record()for item in components]
    diagnostics['native_log_weights']=[float(value)for value in logs]
    return PreparedMomentObservation(dict(observation),tuple(components),diagnostics,prepared)
