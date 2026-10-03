"""
Unit tests for the decision rule: probability -> risk band -> underwriting action.

    probability <  REVIEW_THRESHOLD   ->  LOW    / APPROVE
    probability <  DECLINE_THRESHOLD  ->  MEDIUM / REVIEW
    otherwise                         ->  HIGH   / DECLINE

The rule is pure (no model needed), so these tests run in milliseconds.
"""

import importlib

import pytest

import app.config as config
import app.model as model_module
from app.model import CreditRiskModel


class TestDecide:
    @pytest.mark.parametrize(
        "probability,band,decision",
        [
            (0.0, "LOW", "APPROVE"),
            (0.2999, "LOW", "APPROVE"),
            (0.30, "MEDIUM", "REVIEW"),
            (0.45, "MEDIUM", "REVIEW"),
            (0.5999, "MEDIUM", "REVIEW"),
            (0.60, "HIGH", "DECLINE"),
            (0.95, "HIGH", "DECLINE"),
            (1.0, "HIGH", "DECLINE"),
        ],
    )
    def test_default_thresholds(self, probability, band, decision):
        assert CreditRiskModel.decide(probability) == {"risk_band": band, "decision": decision}

    def test_every_score_above_decline_is_declined(self):
        """Guards the comparison order: checking REVIEW first would stop here."""
        for probability in (0.6, 0.7, 0.8, 0.9, 1.0):
            assert CreditRiskModel.decide(probability)["decision"] == "DECLINE"

    def test_follows_the_configured_thresholds(self, monkeypatch):
        """Thresholds are business settings: moving them moves the decision."""
        monkeypatch.setattr(model_module, "REVIEW_THRESHOLD", 0.10)
        monkeypatch.setattr(model_module, "DECLINE_THRESHOLD", 0.20)
        assert CreditRiskModel.decide(0.05)["decision"] == "APPROVE"
        assert CreditRiskModel.decide(0.15)["decision"] == "REVIEW"
        assert CreditRiskModel.decide(0.25)["decision"] == "DECLINE"


class TestThresholdConfig:
    def test_defaults_match_the_brief(self, monkeypatch):
        monkeypatch.delenv("REVIEW_THRESHOLD", raising=False)
        monkeypatch.delenv("DECLINE_THRESHOLD", raising=False)
        try:
            reloaded = importlib.reload(config)
            assert (reloaded.REVIEW_THRESHOLD, reloaded.DECLINE_THRESHOLD) == (0.30, 0.60)
        finally:
            monkeypatch.undo()
            importlib.reload(config)

    def test_thresholds_are_read_from_the_environment(self, monkeypatch):
        """Same image, different thresholds: only the environment changes."""
        monkeypatch.setenv("REVIEW_THRESHOLD", "0.25")
        monkeypatch.setenv("DECLINE_THRESHOLD", "0.75")
        try:
            reloaded = importlib.reload(config)
            assert (reloaded.REVIEW_THRESHOLD, reloaded.DECLINE_THRESHOLD) == (0.25, 0.75)
        finally:
            monkeypatch.undo()
            importlib.reload(config)
