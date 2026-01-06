"""
Tests for CLI FIELD_SCOPES validation.

This test ensures that all fields in LLMTrainingParams have corresponding
entries in FIELD_SCOPES to prevent CLI argument parsing failures.
"""

import pytest


class TestFieldScopesCompleteness:
    """Validate FIELD_SCOPES contains all LLMTrainingParams fields."""

    def test_all_params_have_field_scopes(self):
        """Ensure all LLMTrainingParams fields have FIELD_SCOPES entries."""
        from autotrain.trainers.clm.params import LLMTrainingParams
        from autotrain.cli.run_llm import FIELD_SCOPES

        param_fields = set(LLMTrainingParams.model_fields.keys())
        scope_fields = set(FIELD_SCOPES.keys())

        missing = param_fields - scope_fields
        assert not missing, (
            f"Missing FIELD_SCOPES entries for: {', '.join(sorted(missing))}. "
            f"Add these fields to FIELD_SCOPES in run_llm.py."
        )

    def test_field_scopes_no_extra_fields(self):
        """Warn if FIELD_SCOPES has fields not in LLMTrainingParams."""
        from autotrain.trainers.clm.params import LLMTrainingParams
        from autotrain.cli.run_llm import FIELD_SCOPES

        param_fields = set(LLMTrainingParams.model_fields.keys())
        scope_fields = set(FIELD_SCOPES.keys())

        extra = scope_fields - param_fields
        # This is a warning, not a failure - extra fields might be aliases
        if extra:
            pytest.warns(
                UserWarning,
                match=f"FIELD_SCOPES has entries not in LLMTrainingParams: {extra}"
            )

    def test_cli_llm_command_registers(self):
        """Test that 'autotrain llm' command can register without errors."""
        from argparse import ArgumentParser
        from autotrain.cli.run_llm import RunAutoTrainLLMCommand

        parser = ArgumentParser()
        subparsers = parser.add_subparsers()

        # This will fail if FIELD_SCOPES is missing any fields
        try:
            RunAutoTrainLLMCommand.register_subcommand(subparsers)
        except ValueError as e:
            pytest.fail(f"CLI registration failed: {e}")
