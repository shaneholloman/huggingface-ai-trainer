"""
Tests for HF repo conflict handling in UploadLogs callback.

Verifies that:
1. First run creates repo normally
2. Second run detects 409, creates versioned repo with datetime suffix
3. Versioned repo_id propagates to config so final push_to_hub uses it
4. Non-409 errors still raise
"""

import re
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from autotrain.trainers.clm.params import LLMTrainingParams


class FakeHfApi:
    """Mock HfApi that tracks create_repo calls and can simulate 409."""

    def __init__(self, existing_repos=None):
        self.existing_repos = set(existing_repos or [])
        self.created_repos = []

    def create_repo(self, repo_id, repo_type="model", private=True):
        if repo_id in self.existing_repos:
            raise Exception(f"409 Conflict: repo {repo_id} already exists")
        self.existing_repos.add(repo_id)
        self.created_repos.append(repo_id)
        return repo_id


class FakePartialState:
    """Mock PartialState that always returns process_index=0."""

    @property
    def process_index(self):
        return 0


def _make_config(repo_id=None):
    """Create a minimal LLMTrainingParams for testing."""
    return LLMTrainingParams(
        model="sshleifer/tiny-gpt2",
        data_path="dummy",
        project_name="/tmp/test-project",
        push_to_hub=True,
        username="testuser",
        token="fake-token",
        repo_id=repo_id,
    )


class TestRepoConflictHandling:
    """Test UploadLogs callback repo conflict detection and versioning."""

    @patch("autotrain.trainers.common.PartialState", FakePartialState)
    @patch("autotrain.trainers.common.HfApi")
    def test_first_run_creates_repo_normally(self, mock_hf_api_cls):
        """First run should create repo with the original name."""
        fake_api = FakeHfApi()
        mock_hf_api_cls.return_value = fake_api

        from autotrain.trainers.common import UploadLogs

        config = _make_config()
        callback = UploadLogs(config)

        assert callback.repo_id == "testuser/test-project"
        assert "testuser/test-project" in fake_api.created_repos
        assert config.repo_id is None  # Not modified when no conflict

    @patch("autotrain.trainers.common.PartialState", FakePartialState)
    @patch("autotrain.trainers.common.HfApi")
    def test_conflict_creates_versioned_repo(self, mock_hf_api_cls):
        """409 conflict should create a versioned repo with datetime suffix."""
        fake_api = FakeHfApi(existing_repos={"testuser/test-project"})
        mock_hf_api_cls.return_value = fake_api

        from autotrain.trainers.common import UploadLogs

        config = _make_config()
        callback = UploadLogs(config)

        # Should NOT be the original name
        assert callback.repo_id != "testuser/test-project"
        # Should match pattern: testuser/test-project-YYYYMMDD-HHMM
        assert re.match(r"testuser/test-project-\d{8}-\d{4}", callback.repo_id)
        # Versioned repo should have been created
        assert callback.repo_id in fake_api.existing_repos

    @patch("autotrain.trainers.common.PartialState", FakePartialState)
    @patch("autotrain.trainers.common.HfApi")
    def test_conflict_propagates_repo_id_to_config(self, mock_hf_api_cls):
        """Versioned repo_id should be set on config so final push_to_hub uses it."""
        fake_api = FakeHfApi(existing_repos={"testuser/test-project"})
        mock_hf_api_cls.return_value = fake_api

        from autotrain.trainers.common import UploadLogs

        config = _make_config()
        assert config.repo_id is None  # Before

        callback = UploadLogs(config)

        # config.repo_id should now be set to the versioned name
        assert config.repo_id is not None
        assert config.repo_id == callback.repo_id
        assert re.match(r"testuser/test-project-\d{8}-\d{4}", config.repo_id)

    @patch("autotrain.trainers.common.PartialState", FakePartialState)
    @patch("autotrain.trainers.common.HfApi")
    def test_conflict_with_explicit_repo_id(self, mock_hf_api_cls):
        """409 with explicit repo_id should still version correctly."""
        fake_api = FakeHfApi(existing_repos={"myorg/my-model"})
        mock_hf_api_cls.return_value = fake_api

        from autotrain.trainers.common import UploadLogs

        config = _make_config(repo_id="myorg/my-model")
        callback = UploadLogs(config)

        # Should have versioned
        assert callback.repo_id != "myorg/my-model"
        assert re.match(r"testuser/my-model-\d{8}-\d{4}", callback.repo_id)
        # Should propagate
        assert config.repo_id == callback.repo_id

    @patch("autotrain.trainers.common.PartialState", FakePartialState)
    @patch("autotrain.trainers.common.HfApi")
    def test_non_409_error_still_raises(self, mock_hf_api_cls):
        """Non-409 errors (auth, network, etc.) should still raise."""
        mock_api = MagicMock()
        mock_api.create_repo.side_effect = Exception("401 Unauthorized")
        mock_hf_api_cls.return_value = mock_api

        from autotrain.trainers.common import UploadLogs

        config = _make_config()
        with pytest.raises(Exception, match="401 Unauthorized"):
            UploadLogs(config)

    @patch("autotrain.trainers.common.PartialState", FakePartialState)
    @patch("autotrain.trainers.common.HfApi")
    def test_final_push_uses_propagated_repo_id(self, mock_hf_api_cls):
        """Simulate that clm/utils.py post_training_steps reads config.repo_id."""
        fake_api = FakeHfApi(existing_repos={"testuser/test-project"})
        mock_hf_api_cls.return_value = fake_api

        from autotrain.trainers.common import UploadLogs

        config = _make_config()
        callback = UploadLogs(config)

        # Simulate what clm/utils.py does for final push:
        if getattr(config, "repo_id", None):
            final_repo_id = config.repo_id
        else:
            final_repo_id = f"{config.username}/test-project"

        # Final push should use the versioned name, not the original
        assert final_repo_id == callback.repo_id
        assert final_repo_id != "testuser/test-project"

    @patch("autotrain.trainers.common.PartialState", FakePartialState)
    @patch("autotrain.trainers.common.HfApi")
    def test_no_conflict_does_not_modify_config(self, mock_hf_api_cls):
        """When there's no conflict, config.repo_id should remain unchanged."""
        fake_api = FakeHfApi()  # No existing repos
        mock_hf_api_cls.return_value = fake_api

        from autotrain.trainers.common import UploadLogs

        config = _make_config()
        original_repo_id = config.repo_id  # None

        callback = UploadLogs(config)

        assert config.repo_id == original_repo_id  # Still None
        assert callback.repo_id == "testuser/test-project"

    @patch("autotrain.trainers.common.PartialState", FakePartialState)
    @patch("autotrain.trainers.common.HfApi")
    def test_versioned_name_has_current_datetime(self, mock_hf_api_cls):
        """Versioned suffix should be close to current time."""
        fake_api = FakeHfApi(existing_repos={"testuser/test-project"})
        mock_hf_api_cls.return_value = fake_api

        from autotrain.trainers.common import UploadLogs

        before = datetime.now().strftime("%Y%m%d-%H%M")
        config = _make_config()
        callback = UploadLogs(config)
        after = datetime.now().strftime("%Y%m%d-%H%M")

        # Extract the suffix
        suffix = callback.repo_id.replace("testuser/test-project-", "")
        # Should be between before and after (usually same minute)
        assert suffix >= before
        assert suffix <= after
