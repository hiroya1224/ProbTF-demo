from threading import RLock
from collections.abc import Mapping

from probtf.distributions import TransformDistributionStamped
from probtf.graph.buffer import EdgeTimeBuffer
from probtf.graph.edge import EdgeView, PhysicalEdge
from probtf.graph.path import PathExpression
from probtf.graph.status import GraphErrorCode, TemporalResolutionError
from probtf.graph.topology import ProbTfTopology
from probtf.temporal import (
    AuthorityConflictPolicy,
    ParentChangePolicy,
    TemporalEvaluationKind,
    TemporalModel,
    TemporalPolicy,
    TemporalQueryMode,
)


class ProbTfGraph:
    """TF-style forest plus timestamped Prob-TF edge histories."""

    def __init__(
        self,
        max_records_per_edge=None,
        authority_conflict_policy=AuthorityConflictPolicy.REJECT,
        parent_change_policy=ParentChangePolicy.REJECT,
        latent_store=None,
    ):
        from probtf.dependency import (
            DependencyAwareMomentEvaluator,
            GaussianLatentStore,
        )

        EdgeTimeBuffer(max_records_per_edge, authority_conflict_policy)
        if latent_store is not None and not isinstance(
            latent_store,
            GaussianLatentStore,
        ):
            raise TypeError("latent_store must be GaussianLatentStore or None.")
        self.topology = ProbTfTopology(parent_change_policy)
        self.max_records_per_edge = max_records_per_edge
        self.authority_conflict_policy = authority_conflict_policy
        self.latent_store = (
            GaussianLatentStore() if latent_store is None else latent_store
        )
        self._dependency_moment_evaluator = DependencyAwareMomentEvaluator()
        self._buffers = {}
        self._temporal_models = {}
        self._default_temporal_models = {}
        self._lock = RLock()

    @property
    def frames(self):
        """Return an immutable snapshot of the currently known frames."""

        with self._lock:
            return self.topology.frames

    @property
    def edges(self):
        """Return an edge-ID-ordered snapshot of known physical edges."""

        with self._lock:
            return tuple(
                self.topology.edge(edge_id)
                for edge_id in sorted(self._buffers)
            )

    def edge_buffer(self, edge_id):
        with self._lock:
            return self._buffers[edge_id]

    @property
    def temporal_model_bindings(self):
        """Return an immutable edge/authority/model-ID ordered snapshot."""

        with self._lock:
            return tuple(
                (edge_id, authority, model_id, model)
                for (edge_id, authority), models in sorted(self._temporal_models.items())
                for model_id, model in sorted(models.items())
            )

    def register_temporal_model(
        self,
        edge_id,
        authority,
        model,
        *,
        make_default=True,
        replace_existing=False,
    ):
        """Bind one named model to one physical edge and one authority."""

        if not isinstance(model, TemporalModel):
            raise TypeError("model must be TemporalModel.")
        if type(make_default) is not bool:
            raise TypeError("make_default must be a built-in bool.")
        if type(replace_existing) is not bool:
            raise TypeError("replace_existing must be a built-in bool.")
        edge_id = str(edge_id).strip()
        authority = str(authority).strip()
        if not edge_id or not authority:
            raise ValueError("edge_id and authority must not be empty.")
        with self._lock:
            if edge_id not in self._buffers:
                raise KeyError("Unknown physical edge_id {!r}.".format(edge_id))
            authorities = {record.authority for record in self._buffers[edge_id].records}
            if authority not in authorities:
                raise KeyError(
                    "Authority {!r} has no records on edge {!r}.".format(
                        authority,
                        edge_id,
                    )
                )
            key = (edge_id, authority)
            models = self._temporal_models.setdefault(key, {})
            if model.model_id in models and not replace_existing:
                raise ValueError(
                    "Temporal model {!r} is already bound to edge {!r}, authority {!r}.".format(
                        model.model_id,
                        edge_id,
                        authority,
                    )
                )
            models[model.model_id] = model
            if make_default:
                self._default_temporal_models[key] = model.model_id
        return model

    def unregister_temporal_model(self, edge_id, authority, model_id):
        edge_id = str(edge_id).strip()
        authority = str(authority).strip()
        model_id = str(model_id).strip()
        key = (edge_id, authority)
        with self._lock:
            models = self._temporal_models.get(key)
            if models is None or model_id not in models:
                raise KeyError((edge_id, authority, model_id))
            model = models.pop(model_id)
            if not models:
                del self._temporal_models[key]
            if self._default_temporal_models.get(key) == model_id:
                self._default_temporal_models.pop(key, None)
            return model

    @staticmethod
    def _edge_model_selector(model_id, edge_id):
        if isinstance(model_id, Mapping):
            selected = model_id.get(edge_id)
        else:
            selected = model_id
        return None if selected is None else str(selected).strip()

    def _selected_temporal_model(self, edge_id, authority, requested_model_id):
        key = (edge_id, authority)
        models = self._temporal_models.get(key, {})
        if requested_model_id is not None:
            return models.get(requested_model_id)
        default_id = self._default_temporal_models.get(key)
        if default_id is not None:
            return models.get(default_id)
        if len(models) == 1:
            return next(iter(models.values()))
        if len(models) > 1:
            raise TemporalResolutionError(
                GraphErrorCode.MODEL_AMBIGUOUS,
                "Multiple named temporal models are bound; specify model_id.",
            )
        return None

    def insert(self, record):
        if not isinstance(record, TransformDistributionStamped):
            raise TypeError("record must be a TransformDistributionStamped.")
        physical = PhysicalEdge(record.edge_id, record.parent_frame_id, record.child_frame_id)
        with self._lock:
            buffer = self._buffers.get(record.edge_id)
            if buffer is None:
                buffer = EdgeTimeBuffer(
                    self.max_records_per_edge,
                    self.authority_conflict_policy,
                )
                buffer.insert(record)
                self.topology.add_edge(physical)
                self._buffers[record.edge_id] = buffer
            else:
                self.topology.add_edge(physical)
                buffer.insert(record)

    def _resolved_traversal(
        self,
        target_frame,
        source_frame,
        stamp,
        policy,
        tolerance,
        max_age,
        model_id,
        max_prediction_horizon,
        random_seed,
        random_stream,
        query_mode,
        max_uncertainty_trace,
        allow_degraded,
        latest_common_model_policy,
    ):
        traversal = self.topology.traversal(source_frame, target_frame)
        if not traversal:
            resolved_stamp = 0.0 if stamp is None else float(stamp)
            return resolved_stamp, ()

        if policy is TemporalPolicy.LATEST_COMMON:
            dynamic_buffers = [
                self._buffers[edge.edge_id]
                for edge, _ in traversal
                if not self._buffers[edge.edge_id].is_static
            ]
            if dynamic_buffers:
                common = min(buffer.latest_stamp for buffer in dynamic_buffers)
                earliest_common = max(buffer.earliest_stamp for buffer in dynamic_buffers)
                if common < earliest_common:
                    raise TemporalResolutionError(
                        GraphErrorCode.TEMPORAL_OUT_OF_RANGE,
                        "Path edge histories have no common availability interval.",
                    )
            else:
                common = 0.0 if stamp is None else float(stamp)
            if latest_common_model_policy is None:
                resolved = tuple(
                    (
                        edge,
                        direction,
                        self._buffers[edge.edge_id].resolve(
                            common,
                            TemporalPolicy.LATEST,
                            max_age=max_age,
                        ),
                    )
                    for edge, direction in traversal
                )
            else:
                if latest_common_model_policy not in (
                    TemporalPolicy.INTERPOLATE_WITH_MODEL,
                    TemporalPolicy.PREDICT_WITH_MODEL,
                ):
                    raise ValueError(
                        "latest_common_model_policy must be a model-based policy or None."
                    )
                entries = []
                for edge, direction in traversal:
                    buffer = self._buffers[edge.edge_id]
                    authority = buffer.model_authority(
                        common,
                        latest_common_model_policy,
                    )
                    selector = self._edge_model_selector(model_id, edge.edge_id)
                    model = self._selected_temporal_model(
                        edge.edge_id,
                        authority,
                        selector,
                    )
                    entries.append(
                        (
                            edge,
                            direction,
                            buffer.resolve(
                                common,
                                latest_common_model_policy,
                                tolerance,
                                max_age,
                                temporal_model=model,
                                model_selector=None if model is None else model.model_id,
                                max_prediction_horizon=max_prediction_horizon,
                                random_seed=random_seed,
                                random_stream=random_stream,
                                query_mode=query_mode,
                                max_uncertainty_trace=max_uncertainty_trace,
                                allow_degraded=allow_degraded,
                            ),
                        )
                    )
                resolved = tuple(entries)
            return common, resolved

        entries = []
        for edge, direction in traversal:
            buffer = self._buffers[edge.edge_id]
            model = None
            selector = self._edge_model_selector(model_id, edge.edge_id)
            if policy in (
                TemporalPolicy.INTERPOLATE_WITH_MODEL,
                TemporalPolicy.PREDICT_WITH_MODEL,
            ):
                authority = buffer.model_authority(stamp, policy)
                model = self._selected_temporal_model(
                    edge.edge_id,
                    authority,
                    selector,
                )
            entries.append(
                (
                    edge,
                    direction,
                    buffer.resolve(
                        stamp,
                        policy,
                        tolerance,
                        max_age,
                        temporal_model=model,
                        model_selector=None if model is None else model.model_id,
                        max_prediction_horizon=max_prediction_horizon,
                        random_seed=random_seed,
                        random_stream=random_stream,
                        query_mode=query_mode,
                        max_uncertainty_trace=max_uncertainty_trace,
                        allow_degraded=allow_degraded,
                    ),
                )
            )
        resolved = tuple(entries)
        if stamp is not None:
            resolved_stamp = float(stamp)
        else:
            resolved_stamp = max(item.sample_stamp for _, _, item in resolved)
        return resolved_stamp, resolved

    def lookup_path(
        self,
        target_frame,
        source_frame,
        stamp=None,
        policy=TemporalPolicy.EXACT,
        tolerance=0.0,
        max_age=None,
        *,
        model_id=None,
        max_prediction_horizon=None,
        random_seed=None,
        random_stream="",
        query_mode=TemporalQueryMode.ONLINE,
        max_uncertainty_trace=None,
        allow_degraded=False,
        latest_common_model_policy=None,
    ):
        if type(allow_degraded) is not bool:
            raise TypeError("allow_degraded must be a built-in bool.")
        with self._lock:
            resolved_stamp, traversal = self._resolved_traversal(
                target_frame,
                source_frame,
                stamp,
                policy,
                tolerance,
                max_age,
                model_id,
                max_prediction_horizon,
                random_seed,
                random_stream,
                query_mode,
                max_uncertainty_trace,
                allow_degraded,
                latest_common_model_policy,
            )
            records = tuple(
                (
                    resolved.record
                    if resolved.evaluation is not None
                    and resolved.evaluation.evaluation_kind
                    in (
                        TemporalEvaluationKind.STATIC,
                        TemporalEvaluationKind.MODEL_INTERPOLATION,
                        TemporalEvaluationKind.MODEL_PREDICTION,
                    )
                    else self._buffers[edge.edge_id].record_at_sample_stamp(
                        resolved.sample_stamp
                    )
                )
                for edge, _, resolved in traversal
            )
            return PathExpression(
                source_frame,
                target_frame,
                resolved_stamp,
                tuple(
                    EdgeView(edge.edge_id, direction, resolved.sample_stamp)
                    for edge, direction, resolved in traversal
                ),
                tuple(
                    "{}:{}".format(
                        edge.edge_id,
                        (
                            "LATEST_COMMON_ZERO_ORDER_HOLD"
                            if policy is TemporalPolicy.LATEST_COMMON
                            and not self._buffers[edge.edge_id].is_static
                            and resolved.sample_stamp != resolved_stamp
                            else resolved.diagnostic
                        ),
                    )
                    for edge, _, resolved in traversal
                    if resolved.diagnostic
                ),
                tuple(resolved.evaluation for _, _, resolved in traversal),
                records,
            )

    def resolved_records(self, path):
        if not isinstance(path, PathExpression):
            raise TypeError("path must be a PathExpression.")
        with self._lock:
            if path._record_snapshot:
                return path._record_snapshot
            return tuple(
                self._buffers[view.edge_id].record_at_sample_stamp(view.sample_stamp)
                for view in path.edge_views
            )

    def lookup_kernel(
        self,
        target_frame,
        source_frame,
        stamp=None,
        policy=TemporalPolicy.EXACT,
        tolerance=0.0,
        max_age=None,
        *,
        model_id=None,
        max_prediction_horizon=None,
        random_seed=None,
        random_stream="",
        query_mode=TemporalQueryMode.ONLINE,
        max_uncertainty_trace=None,
        allow_degraded=False,
        latest_common_model_policy=None,
    ):
        from probtf.kernels import kernel_from_path

        with self._lock:
            path = self.lookup_path(
                target_frame,
                source_frame,
                stamp,
                policy,
                tolerance,
                max_age,
                model_id=model_id,
                max_prediction_horizon=max_prediction_horizon,
                random_seed=random_seed,
                random_stream=random_stream,
                query_mode=query_mode,
                max_uncertainty_trace=max_uncertainty_trace,
                allow_degraded=allow_degraded,
                latest_common_model_policy=latest_common_model_policy,
            )
            return kernel_from_path(path, self.resolved_records(path))

    def lookup_transform_moments(
        self,
        target_frame,
        source_frame,
        stamp=None,
        policy=TemporalPolicy.EXACT,
        tolerance=0.0,
        max_age=None,
        *,
        model_id=None,
        max_prediction_horizon=None,
        random_seed=None,
        random_stream="",
        query_mode=TemporalQueryMode.ONLINE,
        max_uncertainty_trace=None,
        allow_degraded=False,
        latest_common_model_policy=None,
        latent_snapshot=None,
    ):
        """Return a dependency-aware local transform mean and 6x6 covariance."""

        kernel = self.lookup_kernel(
            target_frame,
            source_frame,
            stamp,
            policy,
            tolerance,
            max_age,
            model_id=model_id,
            max_prediction_horizon=max_prediction_horizon,
            random_seed=random_seed,
            random_stream=random_stream,
            query_mode=query_mode,
            max_uncertainty_trace=max_uncertainty_trace,
            allow_degraded=allow_degraded,
            latest_common_model_policy=latest_common_model_policy,
        )
        selected = (
            self.latent_store.snapshot()
            if latent_snapshot is None
            else latent_snapshot
        )
        return self._dependency_moment_evaluator.evaluate(kernel, selected)
