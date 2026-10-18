"""One continuous stiffness update from one declared joint-body batch.

This changes the observation partition, not the retained parameter law into a
point/Gaussian. An aggregate hazard records the exact probability of at least
one change; replacing its latent time by a reset before the entire batch is a
separate, explicit approximation.
"""
from dataclasses import dataclass
import hashlib
import json
import math


def aggregate_event_hazard(probability, count):
    """P(at least one change) for independent equal-probability event trials."""
    probability = float(probability)
    if not math.isfinite(probability) or not 0. <= probability <= 1.:
        raise ValueError('An event change probability must lie in [0, 1].')
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError('An effective event count must be a nonnegative integer.')
    if not count or probability == 0.:
        return 0.
    if probability == 1.:
        return 1.
    return -math.expm1(count*math.log1p(-probability))


def aggregate_event_hazard_logs(probability, count):
    """Keep finite survival even when the ordinary reset probability rounds to 1.

    JSON ``None`` denotes an exact zero-mass branch, i.e. log probability -inf.
    """
    probability = float(probability)
    rho = aggregate_event_hazard(probability, count)
    if not count or probability == 0.:
        return rho, 0., None
    if probability == 1.:
        return rho, None, 0.
    log_keep = count*math.log1p(-probability)
    return rho, log_keep, math.log(-math.expm1(log_keep))


def _read(event, name, default=None):
    return event.get(name, default) if isinstance(event, dict) else getattr(event, name, default)


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError('A joint batch needs finite JSON source identities and ledger records.') from error


@dataclass(frozen=True)
class JointBatchStiffnessContext:
    """Immutable identities/policy; record() returns a fresh JSON mapping."""
    serialized_record: str

    @property
    def batch_id(self):
        return self.record()['batch_id']

    @property
    def observation_ids(self):
        return tuple(self.record()['effective_observation_ids'])

    @property
    def rho_batch(self):
        return self.record()['stiffness_change_probability_batch']

    @property
    def hazard_logs(self):
        record = self.record()
        return (record['stiffness_log_keep_probability'], record['stiffness_log_reset_probability'])

    @property
    def source_digest(self):
        return hashlib.sha256(self.serialized_record.encode()).hexdigest()

    def record(self):
        return json.loads(self.serialized_record)

    def validate_result(self, result):
        record = self.record()
        source = result.prior
        if (source.state_id != record['comparison_source_state_id']
                or float(source.stamp) != record['comparison_source_stamp']
                or tuple(source.command) != tuple(record['comparison_source_command'])
                or source.version != record['comparison_source_version']):
            raise ValueError('A stiffness batch must use its exact frozen body comparison source.')
        if tuple(result.records.get('observation_ids', ())) != self.observation_ids:
            raise ValueError('A stiffness batch must retain every effective observation identity in order.')


def make_joint_batch_stiffness_context(result, *, events, command_ledger,
                                      probability, enabled=True, batch_id=None):
    """Capture all acquisitions while counting only effective sensor factors.

    ``command_ledger`` is the exact existing path record or its checked compact
    journal reference. Raw samples themselves remain in their immutable DB;
    their IDs, payload hashes, acquisition commands and times are retained here.
    """
    events = tuple(events)
    if not events:
        raise ValueError('A joint stiffness batch cannot have no acquired events.')
    if not isinstance(enabled, bool):
        raise ValueError('A stiffness batch hazard enabled flag must be boolean.')
    observed = tuple(result.records.get('observation_ids', ()))
    if len(observed) != len(set(observed)):
        raise ValueError('A joint batch contains duplicate effective observation IDs.')
    event_records, identities, order = [], set(), []
    for event in events:
        identity, kind = _read(event, 'event_id'), _read(event, 'kind')
        sequence, stamp = _read(event, 'sequence'), float(_read(event, 'stamp'))
        command = tuple(float(value) for value in _read(event, 'command', ()))
        if (not isinstance(identity, str) or not identity or identity in identities
                or kind not in ('joint', 'imu', 'image')
                or isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0
                or not math.isfinite(stamp) or not all(math.isfinite(value) for value in command)
                or len(command) != len(result.prior.command)):
            raise ValueError('Joint batch acquisition identities, order and commands must be valid and distinct.')
        if identity in observed and kind == 'joint':
            raise ValueError('A sent-command notification cannot become a stiffness observation.')
        identities.add(identity)
        order.append((stamp, sequence))
        event_records.append(dict(event_id=identity, kind=kind, sequence=sequence,
            stamp=stamp, command=list(command), payload_hash=_read(event, 'payload_hash'),
            used_in_joint_likelihood=identity in observed))
    if order != sorted(order) or len(set(order)) != len(order):
        raise ValueError('A joint batch must preserve acquisition stamp and sequence order.')
    if set(observed)-identities:
        raise ValueError('A joint stiffness batch lost an effective raw observation identity.')
    acquired_observed = tuple(item['event_id'] for item in event_records if item['used_in_joint_likelihood'])
    if acquired_observed != observed:
        raise ValueError('Effective stiffness observations must follow the acquired batch order.')
    if command_ledger is None:
        raise ValueError('A joint stiffness batch requires its actual executed command ledger record.')
    if hasattr(command_ledger, 'record'):
        command_ledger = command_ledger.record()
    ledger_json = _canonical(command_ledger)
    probability = float(probability)
    configured, log_keep, log_reset = aggregate_event_hazard_logs(probability, len(observed))
    source = result.prior
    record = dict(schema='joint_batch_stiffness_v1',
        acquired_events=event_records, acquired_event_ids=[item['event_id'] for item in event_records],
        acquired_event_count=len(event_records), effective_observation_ids=list(observed),
        effective_sensor_event_count=len(observed),
        acquisition_stamp_range=[order[0][0], order[-1][0]],
        comparison_source_state_id=source.state_id, comparison_source_stamp=float(source.stamp),
        comparison_source_command=list(source.command), comparison_source_version=source.version,
        command_ledger=json.loads(ledger_json), command_ledger_sha256=hashlib.sha256(ledger_json.encode()).hexdigest(),
        hazard_enabled=enabled, stiffness_change_probability_event=probability,
        stiffness_change_probability_batch=configured if enabled else 0.,
        stiffness_log_keep_probability=log_keep if enabled else 0.,
        stiffness_log_reset_probability=log_reset if enabled else None,
        zero_probability_log_representation='null means negative infinity; finite log weights are authoritative even if ordinary probability rounds to zero or one',
        hazard_aggregation='rho_batch = 1 - (1-rho_event)^N; N counts effective sensor events, not joint notifications or missing images',
        at_least_one_change_probability_exact_under_declared_event_clock=True,
        reset_time_approximation='conditional on at least one change, reset the continuous stiffness to the existing initial log-K reset measure before the entire joint batch; latent within-batch change times are not retained',
        exact_sequential_hazard_recursion_preserved=False,
        factor_semantics='one full-joint body posterior / actual Keep-Reset body prior density ratio for the joint batch, evaluated by the existing auxiliary FK at the comparison command',
        maximum_new_continuous_factors=1 if observed else 0,
        stiffness_bfs_updates_requested=1 if observed and enabled else 0,
        intermediate_observation_Laplace_updates=False,
        intermediate_command_proposals=False,
        parameter_density_projection_applied=False, previous_batch_factors_pruned=False,
        raw_events_discarded=False, monte_carlo_used=False)
    if batch_id is None:
        batch_id = 'joint_batch:'+hashlib.sha256(_canonical(record).encode()).hexdigest()
    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError('A joint stiffness batch needs a nonempty immutable batch identity.')
    record['batch_id'] = batch_id
    context = JointBatchStiffnessContext(_canonical(record))
    context.validate_result(result)
    return context
