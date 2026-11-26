import pytest

from alpha_evidence_lab.data import synthetic_paths
from alpha_evidence_lab.models import StudyPlan


@pytest.fixture
def plan():
    return StudyPlan(
        data_seeds=(11, 23),
        episodes=2,
        train_sessions=40,
        validation_sessions=10,
        holdout_sessions=10,
        budget=2,
        bootstrap_samples=199,
    )


@pytest.fixture
def paths(plan):
    return synthetic_paths(plan)
