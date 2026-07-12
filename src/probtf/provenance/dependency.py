"""Latent-dependency helpers used by path evaluators."""


def dependency_ids(edge_records):
    identifiers = []
    for record in edge_records:
        identifiers.append(record.edge_id)
        identifiers.extend(record.provenance.derived_from_edge_ids)
    return tuple(identifiers)


def repeated_dependency_ids(edge_records):
    seen = set()
    repeated = []
    for identifier in dependency_ids(edge_records):
        if identifier in seen and identifier not in repeated:
            repeated.append(identifier)
        seen.add(identifier)
    return tuple(repeated)

