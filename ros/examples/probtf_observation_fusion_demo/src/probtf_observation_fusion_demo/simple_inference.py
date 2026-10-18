"""One retained joint body/environment law for the minimal reference loop.

Observation likelihood construction and the existing variational Gaussian
update are unchanged. Native observation modes are fused before one joint
moment approximation. Optional body-only surprise uses the declared reset
kernel; there is no separate fresh goal or Schmidt goal retention.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import rospkg
import yaml

import numpy as np

from probtf.spherical_law.fusion import PosteriorComponent
from .bayes_factor import projection_input
from .body_projection import project_update
from .model import FrozenBodyBranch, Stage, WeightedState, _joint_record
from .moving_inference import MovingInference


class SimpleInference(MovingInference):
    """MovingInference acquisition API with a single canonical joint state."""

    def __init__(self, robot_description, observation_templates, config_path=None, *,
                 body_surprise_enabled=None):
        super().__init__(robot_description, observation_templates, config_path=config_path)
        self.law = deepcopy(self.law)
        for edge in self.law['edges']:
            if edge['child'] in self.body_ids:
                edge['translation']['covariance'] = np.zeros((3, 3)).tolist()
        self.law['law_id'] += ':simple_reference_known_local_translations'
        self.settings = deepcopy(self.settings)
        package = Path(rospkg.RosPack().get_path('probtf_observation_fusion_demo'))
        simple_settings = yaml.safe_load((package/'config/simple_reference.yaml').read_text())
        self.settings['simple_startup'] = dict(simple_settings.get('startup', {}))
        self.settings['simple_reference'] = dict(simple_settings['selection'])
        self.settings['simple_stiffness'] = dict(simple_settings['stiffness'])
        self.settings['observation_update'].update(simple_settings.get('observation_update', {}))
        self.settings['async_execution']['observation_selection'] = dict(simple_settings['observation_selection'])
        self.surprise_settings = dict(enabled=True, change_probability=.25,
            event_probability_unit='per_effective_sensor_event')
        self.surprise_settings.update(simple_settings.get('body_bayes_factor_surprise', {}))
        if body_surprise_enabled is not None:
            if not isinstance(body_surprise_enabled, (bool, np.bool_)):
                raise ValueError('body_surprise_enabled must be a boolean or None.')
            self.surprise_settings['enabled'] = bool(body_surprise_enabled)
        if not isinstance(self.surprise_settings['enabled'], (bool, np.bool_)):
            raise ValueError('Body Bayes-factor enabled setting must be a boolean.')
        self.surprise_enabled = bool(self.surprise_settings['enabled'])
        self.change_probability = float(self.surprise_settings['change_probability'])
        if not np.isfinite(self.change_probability) or not 0. <= self.change_probability < 1.:
            raise ValueError('Body Bayes-factor change probability must lie in [0, 1).')
        self.settings['body_bayes_factor_surprise'] = dict(self.surprise_settings)
        self.settings['bayes_factor_surprise'] = dict(self.surprise_settings)
        self.settings['control']['mode'] = 'simple_reference'
        self.settings['control']['visibility_factor']['enabled'] = False
        self.settings['control']['local_quadratic_model']['enabled'] = False
        self.settings['control']['visual_approach']['enabled'] = False
        self.settings['control']['marker_search']['enabled'] = False
        self.common['control'] = self.settings['control']
        self.settings['observation_update']['method'] = 'variational_moments'
        self.observation_update_method = 'variational_moments'
        self.observation_update_settings = dict(self.settings['observation_update'])
        self.common['prior']['independence'] = (
            'declared independent initial law only; retain one joint body/environment posterior and every cross covariance thereafter')
        self.common['prior']['body_translation'] = 'known URDF local attachment translations represented by Dirac measures'
        self.goal_observed = False

    def _project_internal_update(self, result, predicted_goals):
        # Goal candidates are never independently transported or conditioned.
        # The environment is part of this same joint posterior.
        if predicted_goals:
            raise ValueError('The simple reference has one joint body/environment law, not independent goal laws.')
        transition = bool(result.records.get('bayes_factor_surprise', {}).get('applied'))
        native_branches = result.frozen_body_branches
        projected_input = projection_input(result) if transition else result
        projection = project_update(projected_input, construct_conditional_goals=False)
        original = native_branches[0]
        raw_stage = result.stages[-1]
        internal = WeightedState(projection.state.component_id, 1., projection.state, 0.)
        canonical = Stage('fused', (internal,), raw_stage.sensor_ids, ())
        child = PosteriorComponent(internal.label, projection.posterior, 1.,
            lineage=tuple(item.label for item in result.frozen_body_posterior.components), log_weight=0.)
        frozen = replace(result.frozen_body_posterior, components=(child,))
        branch = FrozenBodyBranch(original.label, 1., original.reference_state,
                                  original.prior, (child,), 0.)
        stages = result.stages[:-1]+((
            replace(raw_stage, label='fused_change_candidates'),
            replace(projected_input.stages[-1], label='fused_observation_candidates'), canonical)
            if transition else (replace(raw_stage, label='fused_observation_candidates'), canonical))
        records = result.records
        if transition:
            records['bayes_factor_transport'] = projected_input.records['bayes_factor_transport']
        records['internal_projection'] = dict(projection.record,
            goal_construction='query the environment variable of the same canonical joint body/environment law',
            goal_state_ids=[projection.state.state_id],
            goal_source_state_ids=[projection.state.state_id],
            conditional_goal_diagnostics_used_for_task=False,
            separate_goal_law_constructed=False)
        records['mixture_policy'] = (
            'all Keep/Reset and native observation likelihood components and masses enter the existing variational update; '
            'one full joint Gaussian/Bingham moment approximation, including environment and all cross blocks')
        records['stages'][-1]['label'] = 'fused_change_candidates' if transition else 'fused_observation_candidates'
        if transition:
            records['stages'].append(dict(label='fused_observation_candidates',
                sensor_ids=list(raw_stage.sensor_ids), components=[_joint_record(
                    item, self.paths, result.root_frame) for item in projected_input.stages[-1].components]))
        records['stages'].append(dict(label=canonical.label, sensor_ids=list(canonical.sensor_ids),
            components=[_joint_record(internal, self.paths, result.root_frame)], goal_components=[]))
        records['frozen_internal_charts'] = [dict(prior_label=branch.label,
            reference_state_id=branch.reference_state.state_id, posterior_labels=[child.label],
            approximation='one full joint Gaussian moment approximation after the native observation update')]
        return replace(result, stages=stages, frozen_body_posterior=frozen,
            frozen_body_branches=native_branches if transition else (branch,),
            projected_body_numerator=branch if transition else None)

    def _record_fixed_goal_memory(self, result):
        """Legacy display metadata only; no independently stored goal law."""
        estimates = tuple(item for item in result.local_estimates
                          if item.posterior is not None and item.detector_result is not None)
        previous = ({} if self.last_result is None else
                    self.last_result.records.get('fixed_goal_memory', {}))
        observed = {}
        if estimates:
            self.goal_observed = True
            source = estimates[0].posterior.query
            observed = dict(observation_stamp=source.stamp, observation_command=list(source.command),
                observation_version=source.version,
                observation_ids=list(dict.fromkeys(item.posterior.observation_id for item in estimates)),
                observation_sensor_ids=list(dict.fromkeys(item.posterior.sensor_id for item in estimates)))
        else:
            observed = {key: previous[key] for key in ('observation_stamp', 'observation_command',
                'observation_version', 'observation_ids', 'observation_sensor_ids') if key in previous}
        state = result.stages[-1].components[0].state
        result.records['fixed_goal_memory'] = dict(available=self.goal_observed,
            root_frame=result.root_frame, metadata_only=True, independent_goal_law=False,
            source_state_id=state.state_id, source_stamp=state.stamp, source_command=list(state.command),
            construction='same joint environment posterior followed by known marker-to-switch attachment',
            construction_approximation='no independent fresh-observation task cut; same canonical joint law',
            **observed)
        result.records['fixed_goal_memory_event'] = (
            'joint_environment_marker_update' if estimates else
            'joint_environment_retained' if self.goal_observed else 'unobserved_prior')
        result.records['simple_goal_observed'] = self.goal_observed
        result.records['simple_inference'] = dict(
            goal_observed=self.goal_observed, body_BFS_enabled=self.surprise_enabled,
            independent_fresh_goal=False, goal_components_independent=False,
            state_id=state.state_id, state_stamp=state.stamp, state_command=list(state.command),
            joint_body_environment_cross_retained=True, local_translations_known_Dirac=True,
            observation_update='existing variational Gaussian/Bingham-derived moment update',
            physical_truth_received=False, monte_carlo_used=False)

    def _record_candidate_measure(self, result):
        super()._record_candidate_measure(result)
        if not result.records.get('bayes_factor_surprise', {}).get('applied'):
            result.records['candidate_measure'].pop('bayes_factor_hypothesis_scale', None)

    def _update_common(self, *args, **kwargs):
        previous = self.last_result
        result = super()._update_common(*args, **kwargs)
        if result is previous:
            return result
        result.records['moving_input_family']['temporal_policy'] = (
            'one joint body/environment posterior, transported by actual command increments; '
            'body-only Keep/Reset hypotheses on fresh valid observations when enabled; '
            'all native observation branches enter each fixed-prior variational update; '
            'reset keeps the current environment marginal and redraws only body variables')
        result.records['bayes_factor_surprise']['scope'] = 'body-only; K Bayes-factor surprise remains disabled'
        return result
