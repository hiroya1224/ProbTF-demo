from enum import Enum


class TemporalPolicy(Enum):
    EXACT = "exact"
    NEAREST_WITHIN_TOLERANCE = "nearest_within_tolerance"
    LATEST = "latest"
    LATEST_COMMON = "latest_common"
    INTERPOLATE_WITH_MODEL = "interpolate_with_model"
    PREDICT_WITH_MODEL = "predict_with_model"


class AuthorityConflictPolicy(Enum):
    REJECT = "reject"
    REPLACE = "replace"
    KEEP_FIRST = "keep_first"


class ParentChangePolicy(Enum):
    REJECT = "reject"
    REPLACE_WITH_DIAGNOSTIC = "replace_with_diagnostic"

