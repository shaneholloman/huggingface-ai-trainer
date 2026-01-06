"""
Tests for Hyperparameter Sweep Utility
=======================================
"""

import json
import os
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from autotrain.utils import (
    HyperparameterSweep,
    ParameterRange,
    SweepBackend,
    SweepConfig,
    SweepResult,
    TrialInfo,
    run_autotrain_sweep,
)


def create_sweep(parameters, train_fn=None, metric="loss", direction="minimize", n_trials=10, **kwargs):
    """Create sweep helper function."""
    # Convert parameters to proper format
    param_configs = {}
    for name, spec in parameters.items():
        if isinstance(spec, list):
            param_configs[name] = spec
        elif isinstance(spec, tuple) and len(spec) == 3:
            low, high, dist = spec
            param_configs[name] = ParameterRange(
                low=low, high=high, distribution=dist if dist != "float" else "uniform"
            )
        else:
            param_configs[name] = spec

    config = SweepConfig(parameters=param_configs, metric=metric, direction=direction, n_trials=n_trials, **kwargs)
    return HyperparameterSweep(config, train_fn)


@pytest.fixture
def parameter_ranges():
    """Create test parameter ranges."""
    return {
        "learning_rate": ParameterRange(low=1e-5, high=1e-3, distribution="log_uniform"),
        "batch_size": [4, 8, 16, 32],
        "dropout": ParameterRange(low=0.0, high=0.5, distribution="uniform"),
        "epochs": ParameterRange(low=1, high=5, distribution="int_uniform"),
    }


@pytest.fixture
def sweep_config(parameter_ranges):
    """Create test sweep configuration."""
    return SweepConfig(
        parameters=parameter_ranges,
        metric="eval_loss",
        direction="minimize",
        n_trials=10,
        backend=SweepBackend.RANDOM_SEARCH,
    )


@pytest.fixture
def mock_train_function():
    """Create mock training function."""

    def train_fn(params):
        # Simulate training with random loss based on learning rate
        lr = params.get("learning_rate", 1e-4)
        loss = np.random.random() * lr * 100
        return loss

    return train_fn


def test_parameter_range():
    """Test ParameterRange sampling."""
    # Float parameter
    param = ParameterRange(low=0.0, high=1.0, distribution="uniform")
    value = param.sample()
    assert 0.0 <= value <= 1.0

    # Int parameter
    param = ParameterRange(low=1, high=10, distribution="int_uniform")
    value = param.sample()
    assert isinstance(value, (int, np.integer))
    assert 1 <= value <= 10

    # Log uniform parameter
    param = ParameterRange(low=1e-5, high=1e-3, distribution="log_uniform")
    value = param.sample()
    assert 1e-5 <= value <= 1e-3


def test_sweep_config():
    """Test SweepConfig creation."""
    config = SweepConfig(
        parameters=[],
        metric="accuracy",
        direction="maximize",
        n_trials=50,
        backend=SweepBackend.OPTUNA,
    )

    assert config.metric == "accuracy"
    assert config.direction == "maximize"
    assert config.n_trials == 50
    assert config.backend == SweepBackend.OPTUNA


def test_sweep_result():
    """Test SweepResult operations."""
    config = SweepConfig(parameters=[], metric="loss", direction="minimize")
    result = SweepResult(config)

    # Add trials
    result.add_trial(0, {"lr": 1e-4, "batch_size": 8}, 0.5)
    result.add_trial(1, {"lr": 1e-3, "batch_size": 16}, 0.3)
    result.add_trial(2, {"lr": 1e-5, "batch_size": 4}, 0.7)

    assert len(result.trials) == 3
    assert result.best_value == 0.3
    assert result.best_trial_id == 1
    assert result.best_params["lr"] == 1e-3

    # Test DataFrame conversion
    df = result.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "lr" in df.columns
    assert "value" in df.columns


def test_sweep_result_maximize():
    """Test SweepResult with maximize direction."""
    config = SweepConfig(parameters=[], metric="accuracy", direction="maximize")
    result = SweepResult(config)

    result.add_trial(0, {"lr": 1e-4}, 0.8)
    result.add_trial(1, {"lr": 1e-3}, 0.9)
    result.add_trial(2, {"lr": 1e-5}, 0.7)

    assert result.best_value == 0.9
    assert result.best_trial_id == 1


def test_sweep_result_save_load(tmp_path):
    """Test saving and loading results."""
    config = SweepConfig(parameters=[], metric="loss")
    result = SweepResult(config)

    result.add_trial(0, {"lr": 1e-4}, 0.5)
    result.add_trial(1, {"lr": 1e-3}, 0.3)

    # Save results
    save_path = tmp_path / "results.json"
    result.save(str(save_path))

    assert save_path.exists()
    csv_path = tmp_path / "results.csv"
    assert csv_path.exists()

    # Load and verify
    with open(save_path, "r") as f:
        data = json.load(f)

    assert data["best_value"] == 0.3
    assert len(data["trials"]) == 2


def test_hyperparameter_sweep_random(sweep_config, mock_train_function):
    """Test HyperparameterSweep with random search."""
    sweep_config.n_trials = 5
    sweep_config.backend = SweepBackend.RANDOM_SEARCH

    with tempfile.TemporaryDirectory() as tmp_dir:
        sweep_config.output_dir = tmp_dir

        sweep = HyperparameterSweep(sweep_config, mock_train_function)
        result = sweep.run()

        assert len(result.trials) == 5
        assert result.best_value is not None
        assert result.best_params is not None

        # Check that output files were created
        assert os.path.exists(os.path.join(tmp_dir, "sweep_results.json"))


def test_hyperparameter_sweep_grid(parameter_ranges, mock_train_function):
    """Test HyperparameterSweep with grid search."""
    # Use smaller parameter space for grid search
    params = [
        ParameterRange(name="lr", param_type="categorical", choices=[1e-4, 1e-3]),
        ParameterRange(name="batch_size", param_type="categorical", choices=[8, 16]),
    ]

    config = SweepConfig(
        parameters=params,
        metric="loss",
        direction="minimize",
        backend=SweepBackend.GRID_SEARCH,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        config.output_dir = tmp_dir

        sweep = HyperparameterSweep(config, mock_train_function)
        result = sweep.run()

        # Grid search should try all 4 combinations
        assert len(result.trials) == 4


@pytest.mark.skipif(not pytest.importorskip("optuna"), reason="Optuna not installed")
def test_hyperparameter_sweep_optuna(sweep_config, mock_train_function):
    """Test HyperparameterSweep with Optuna backend."""
    sweep_config.n_trials = 5
    sweep_config.backend = SweepBackend.OPTUNA
    sweep_config.sampler = "tpe"

    with tempfile.TemporaryDirectory() as tmp_dir:
        sweep_config.output_dir = tmp_dir

        sweep = HyperparameterSweep(sweep_config, mock_train_function)
        result = sweep.run()

        assert len(result.trials) == 5
        assert result.study is not None  # Optuna study object


def test_create_sweep():
    """Test create_sweep convenience function."""
    parameters = {
        "learning_rate": (1e-5, 1e-3, "log_uniform"),
        "batch_size": [4, 8, 16],
        "dropout": (0.0, 0.5, "float"),
    }

    def train_fn(params):
        return np.random.random()

    sweep = create_sweep(
        parameters,
        train_fn,
        metric="loss",
        n_trials=10,
    )

    assert isinstance(sweep, HyperparameterSweep)
    assert len(sweep.config.parameters) == 3
    assert sweep.config.n_trials == 10


def test_parameter_range_with_step():
    """Test parameter range with step."""
    param = ParameterRange(
        name="dropout",
        param_type="float",
        low=0.0,
        high=0.5,
        step=0.1,
    )

    # For grid search, step should be used
    values = []
    for _ in range(100):
        value = param.sample(trial=None, backend=SweepBackend.RANDOM_SEARCH)
        values.append(value)

    assert all(0.0 <= v <= 0.5 for v in values)


def test_sweep_early_stopping():
    """Test early stopping in sweep."""
    config = SweepConfig(
        parameters=[],
        metric="loss",
        direction="minimize",
        n_trials=100,  # High number
        patience=3,
        min_delta=0.001,
        backend=SweepBackend.GRID_SEARCH,
    )

    # Create function that gets worse over time
    trial_count = [0]

    def train_fn(params):
        trial_count[0] += 1
        if trial_count[0] <= 2:
            return 0.1  # Good initial values
        else:
            return 0.9  # Bad values trigger early stopping

    with tempfile.TemporaryDirectory() as tmp_dir:
        config.output_dir = tmp_dir

        # Create dummy parameter for grid search
        config.parameters = [ParameterRange(name="dummy", param_type="int", low=1, high=20)]

        sweep = HyperparameterSweep(config, train_fn)
        result = sweep.run()

        # Should stop early, not run all 20 trials
        assert len(result.trials) < 20


def test_sweep_result_plotting(tmp_path):
    """Test result plotting methods."""
    config = SweepConfig(parameters=[], metric="loss", output_dir=str(tmp_path))
    result = SweepResult(config)

    # Add some trials
    for i in range(10):
        result.add_trial(i, {"lr": 10 ** (-5 + i * 0.2)}, 0.5 - i * 0.05)

    # Test optimization history plot
    plot_path = tmp_path / "opt_history.png"
    result.plot_optimization_history(str(plot_path))
    # Just check it doesn't crash - actual plot checking would require image comparison

    # Test parallel coordinates plot (requires plotly)
    try:
        import plotly

        html_path = tmp_path / "parallel.html"
        result.plot_parallel_coordinates(str(html_path))
    except ImportError:
        pass


def test_run_autotrain_sweep():
    """Test AutoTrain integration function."""
    model_config = {
        "model": "gpt2",
        "learning_rate": 1e-4,
        "train_batch_size": 8,
    }

    sweep_parameters = {
        "learning_rate": (1e-5, 1e-3, "log_uniform"),
        "train_batch_size": [4, 8, 16],
    }

    def mock_train(params):
        return np.random.random()

    with tempfile.TemporaryDirectory() as tmp_dir:
        result = run_autotrain_sweep(
            model_config,
            sweep_parameters,
            train_function=mock_train,
            n_trials=5,
            backend="random_search",
            output_dir=tmp_dir,
        )

        assert isinstance(result, SweepResult)
        assert len(result.trials) == 5


def test_run_autotrain_sweep_dict_format():
    """Test AutoTrain sweep with dict format parameters (e.g., from JSON)."""
    model_config = {
        "model": "gpt2",
        "learning_rate": 1e-4,
        "train_batch_size": 8,
    }

    # Dict format as documented: {"type": "categorical", "values": [...]}
    sweep_parameters = {
        "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-3},
        "train_batch_size": {"type": "categorical", "values": [4, 8, 16]},
        "warmup_ratio": {"type": "uniform", "low": 0.0, "high": 0.2},
        "epochs": {"type": "int", "low": 1, "high": 5},
    }

    def mock_train(params):
        # Verify params are correctly sampled
        assert "learning_rate" in params
        assert "train_batch_size" in params
        assert "warmup_ratio" in params
        assert "epochs" in params
        # Check batch_size is from the categorical values
        assert params["train_batch_size"] in [4, 8, 16]
        # Check learning_rate is in log-uniform range
        assert 1e-5 <= params["learning_rate"] <= 1e-3
        # Check warmup_ratio is in uniform range
        assert 0.0 <= params["warmup_ratio"] <= 0.2
        # Check epochs is an integer in range
        assert 1 <= params["epochs"] <= 5
        return np.random.random()

    with tempfile.TemporaryDirectory() as tmp_dir:
        result = run_autotrain_sweep(
            model_config,
            sweep_parameters,
            train_function=mock_train,
            n_trials=5,
            backend="random",
            output_dir=tmp_dir,
        )

        assert isinstance(result, SweepResult)
        assert len(result.trials) == 5


def test_sweep_parallel_trials():
    """Test parallel trial execution."""
    config = SweepConfig(
        parameters=[ParameterRange(name="x", param_type="float", low=0, high=1)],
        n_trials=10,
        n_jobs=2,  # Parallel jobs
        backend=SweepBackend.RANDOM_SEARCH,
    )

    def slow_train_fn(params):
        import time

        time.sleep(0.01)  # Simulate training time
        return params["x"] ** 2

    with tempfile.TemporaryDirectory() as tmp_dir:
        config.output_dir = tmp_dir

        sweep = HyperparameterSweep(config, slow_train_fn)

        # This should be faster with n_jobs=2 than n_jobs=1
        # Just test that it completes without error
        result = sweep.run()
        assert len(result.trials) == 10


def test_sweep_with_failed_trials():
    """Test sweep handling failed trials."""
    config = SweepConfig(
        parameters=[ParameterRange(name="x", param_type="float", low=0, high=1)],
        n_trials=5,
        backend=SweepBackend.RANDOM_SEARCH,
    )

    def failing_train_fn(params):
        if params["x"] > 0.5:
            raise ValueError("Simulated failure")
        return params["x"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        config.output_dir = tmp_dir

        sweep = HyperparameterSweep(config, failing_train_fn)

        # Should handle failures gracefully
        with patch("autotrain.logger.error") as mock_error:
            result = sweep.run()

            # Some trials should have failed
            assert mock_error.called
            # But we should still have some successful trials
            assert len(result.trials) > 0
            assert result.best_value is not None


@pytest.mark.skipif(not pytest.importorskip("optuna"), reason="Optuna not installed")
def test_wandb_sweep_integration():
    """Test W&B native sweep integration creates sweep and links runs."""
    config = SweepConfig(
        parameters=[ParameterRange(name="lr", param_type="float", low=1e-5, high=1e-3)],
        n_trials=2,
        backend=SweepBackend.OPTUNA,
        wandb_sweep=True,
        wandb_project="test-project",
        wandb_entity="test-entity",
    )

    def train_fn(params):
        return params["lr"] * 100

    with patch("wandb.sweep") as mock_wandb_sweep, \
         patch("wandb.init") as mock_wandb_init, \
         patch("wandb.finish") as mock_wandb_finish, \
         patch("wandb.run", None):

        mock_wandb_sweep.return_value = "test-sweep-id-123"

        sweep = HyperparameterSweep(config, train_fn)
        result = sweep.run()

        # Verify wandb.sweep was called with correct config
        mock_wandb_sweep.assert_called_once()
        call_kwargs = mock_wandb_sweep.call_args[1]
        assert call_kwargs["project"] == "test-project"
        assert call_kwargs["entity"] == "test-entity"
        sweep_config_arg = call_kwargs["sweep"]
        assert sweep_config_arg["method"] == "bayes"  # optuna maps to bayes
        assert sweep_config_arg["metric"]["goal"] == "minimize"
        assert "lr" in sweep_config_arg["parameters"]

        # Verify wandb.init was called for each trial with group=sweep_id
        assert mock_wandb_init.call_count == 2
        for call in mock_wandb_init.call_args_list:
            call_kwargs = call[1]
            assert call_kwargs["group"] == "test-sweep-id-123"
            assert call_kwargs["project"] == "test-project"
            assert call_kwargs["entity"] == "test-entity"

        # Verify sweep_id is attached to result
        assert result.wandb_sweep_id == "test-sweep-id-123"
        assert result.wandb_project == "test-project"


def test_wandb_sweep_config_passed_to_run_autotrain_sweep():
    """Test W&B params are passed through run_autotrain_sweep."""
    sweep_parameters = {"lr": (1e-5, 1e-3, "log_uniform")}

    def mock_train(params):
        return 0.5

    with patch("autotrain.utils.HyperparameterSweep") as MockSweep:
        mock_instance = Mock()
        mock_instance.run.return_value = SweepResult(best_params={"lr": 1e-4}, best_value=0.5, trials=[])
        MockSweep.return_value = mock_instance

        run_autotrain_sweep(
            model_config={},
            sweep_parameters=sweep_parameters,
            train_function=mock_train,
            n_trials=2,
            backend="optuna",
            wandb_sweep=True,
            wandb_project="my-project",
            wandb_entity="my-team",
        )

        # Verify SweepConfig was created with W&B params
        call_args = MockSweep.call_args[0][0]  # First positional arg is config
        assert call_args.wandb_sweep is True
        assert call_args.wandb_project == "my-project"
        assert call_args.wandb_entity == "my-team"


def _is_wandb_available_and_logged_in():
    """Check if wandb is installed and logged in."""
    try:
        import wandb
        return wandb.api.api_key is not None
    except Exception:
        return False


@pytest.mark.skipif(
    not _is_wandb_available_and_logged_in(),
    reason="W&B not available or not logged in (run: wandb login)"
)
class TestWandbSweepRealIntegration:
    """Real W&B integration tests - only run when logged in."""

    def test_wandb_sweep_creates_real_sweep(self):
        """Test that W&B sweep is created and runs are linked in real W&B."""
        import time

        config = SweepConfig(
            parameters={
                "lr": ParameterRange(low=1e-5, high=1e-3, distribution="log_uniform"),
                "batch_size": [2, 4],
            },
            n_trials=2,
            backend=SweepBackend.OPTUNA,
            direction="minimize",
            metric="loss",
            wandb_sweep=True,
            wandb_project="aitraining-pytest-sweep",
        )

        def mock_train(params):
            time.sleep(0.1)
            return params.get("lr", 1e-4) * 100

        sweep = HyperparameterSweep(config, mock_train)
        result = sweep.run()

        # Verify sweep was created
        assert hasattr(result, "wandb_sweep_id")
        assert result.wandb_sweep_id is not None
        assert len(result.wandb_sweep_id) > 0
        assert result.wandb_project == "aitraining-pytest-sweep"
        assert len(result.trials) == 2

    def test_wandb_sweep_continue_existing(self):
        """Test continuing an existing W&B sweep."""
        import time

        # Create initial sweep
        config1 = SweepConfig(
            parameters={"lr": [1e-4, 1e-3]},
            n_trials=1,
            backend=SweepBackend.OPTUNA,
            wandb_sweep=True,
            wandb_project="aitraining-pytest-sweep",
        )

        def mock_train(params):
            time.sleep(0.1)
            return 0.5

        sweep1 = HyperparameterSweep(config1, mock_train)
        result1 = sweep1.run()

        assert result1.wandb_sweep_id is not None
        sweep_id = result1.wandb_sweep_id

        # Continue with same sweep_id
        config2 = SweepConfig(
            parameters={"lr": [1e-4, 1e-3]},
            n_trials=1,
            backend=SweepBackend.OPTUNA,
            wandb_sweep=True,
            wandb_project="aitraining-pytest-sweep",
            wandb_sweep_id=sweep_id,  # Continue existing
        )

        sweep2 = HyperparameterSweep(config2, mock_train)
        result2 = sweep2.run()

        # Should use the same sweep_id
        assert result2.wandb_sweep_id == sweep_id


class TestWandbProjectBasename:
    """Tests for WANDB_PROJECT using basename instead of full path."""

    def test_wandb_project_uses_basename_when_project_name_is_path(self):
        """Test that WANDB_PROJECT is set to basename when project_name is a path."""
        from autotrain.trainers.clm.params import LLMTrainingParams
        from autotrain.trainers.clm.sweep_utils import run_with_sweep

        # Create config with project_name as a path (like output directory)
        config = LLMTrainingParams(
            model="test-model",
            project_name="/workspace/trainings/hotel-sft-optuna-v2",
            log="wandb",
            use_sweep=True,
            sweep_backend="optuna",
            sweep_trials=1,
        )

        # Track what WANDB_PROJECT gets set to
        captured_project = None

        def mock_train_func(trial_config):
            nonlocal captured_project
            captured_project = os.environ.get("WANDB_PROJECT")
            # Return a mock trainer with minimal state
            mock_trainer = Mock()
            mock_trainer.state.log_history = [{"eval_loss": 0.5}]
            return mock_trainer

        # Mock run_autotrain_sweep to just call our train function once
        with patch("autotrain.utils.run_autotrain_sweep") as mock_sweep:
            # Simulate sweep calling the train function
            def run_sweep_side_effect(**kwargs):
                train_fn = kwargs["train_function"]
                train_fn({"lr": 1e-4})  # Call with dummy params
                return Mock(best_params={"lr": 1e-4}, best_value=0.5, trials=[])

            mock_sweep.side_effect = run_sweep_side_effect

            run_with_sweep(config, mock_train_func)

        # Verify WANDB_PROJECT was set to basename, not full path
        assert captured_project == "hotel-sft-optuna-v2", (
            f"Expected WANDB_PROJECT='hotel-sft-optuna-v2', got '{captured_project}'"
        )

    def test_wandb_project_uses_explicit_sweep_project_over_basename(self):
        """Test that explicit wandb_sweep_project takes precedence over basename."""
        from autotrain.trainers.clm.params import LLMTrainingParams
        from autotrain.trainers.clm.sweep_utils import run_with_sweep

        config = LLMTrainingParams(
            model="test-model",
            project_name="/workspace/trainings/hotel-sft-optuna-v2",
            log="wandb",
            use_sweep=True,
            sweep_backend="optuna",
            sweep_trials=1,
            wandb_sweep_project="my-custom-project",  # Explicit project name
        )

        captured_project = None

        def mock_train_func(trial_config):
            nonlocal captured_project
            captured_project = os.environ.get("WANDB_PROJECT")
            mock_trainer = Mock()
            mock_trainer.state.log_history = [{"eval_loss": 0.5}]
            return mock_trainer

        with patch("autotrain.utils.run_autotrain_sweep") as mock_sweep:
            def run_sweep_side_effect(**kwargs):
                train_fn = kwargs["train_function"]
                train_fn({"lr": 1e-4})
                return Mock(best_params={"lr": 1e-4}, best_value=0.5, trials=[])

            mock_sweep.side_effect = run_sweep_side_effect

            run_with_sweep(config, mock_train_func)

        # Should use explicit wandb_sweep_project, not basename
        assert captured_project == "my-custom-project", (
            f"Expected WANDB_PROJECT='my-custom-project', got '{captured_project}'"
        )


class TestPostTrialCallback:
    """Tests for post-trial callback and script functionality."""

    def test_post_trial_callback_invoked(self):
        """Test that post_trial_callback is called after each trial."""
        from autotrain.utils import TrialInfo

        callback_calls = []

        def my_callback(trial_info: TrialInfo):
            callback_calls.append(trial_info)

        config = SweepConfig(
            parameters={"x": [0.1, 0.2, 0.3]},
            n_trials=3,
            backend=SweepBackend.RANDOM_SEARCH,
            post_trial_callback=my_callback,
        )

        def train_fn(params):
            return params["x"] * 10

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.output_dir = tmp_dir
            sweep = HyperparameterSweep(config, train_fn)
            result = sweep.run()

            # Callback should be called once per trial
            assert len(callback_calls) == 3
            # Each call should receive a TrialInfo object
            for info in callback_calls:
                assert isinstance(info, TrialInfo)
                assert info.trial_number >= 0
                assert "x" in info.params
                assert info.metric_value is not None

    def test_post_trial_callback_receives_correct_info(self):
        """Test that callback receives correct TrialInfo with is_best tracking."""
        from autotrain.utils import TrialInfo

        callback_calls = []

        def my_callback(trial_info: TrialInfo):
            callback_calls.append(trial_info)

        config = SweepConfig(
            parameters={"x": [0.5, 0.1, 0.3]},  # 0.1 will be best (minimize)
            n_trials=3,
            backend=SweepBackend.GRID_SEARCH,
            direction="minimize",
            post_trial_callback=my_callback,
        )

        def train_fn(params):
            return params["x"]  # Lower is better

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.output_dir = tmp_dir
            sweep = HyperparameterSweep(config, train_fn)
            result = sweep.run()

            assert len(callback_calls) == 3

            # Find which call had is_best=True
            best_calls = [c for c in callback_calls if c.is_best]
            # At least one should be marked as best
            assert len(best_calls) >= 1
            # The best one should have the lowest metric value
            best_call = best_calls[-1]  # Last one marked as best is the actual best
            assert best_call.metric_value == min(c.metric_value for c in callback_calls)

    def test_post_trial_callback_maximize_direction(self):
        """Test callback with maximize direction correctly identifies best."""
        from autotrain.utils import TrialInfo

        callback_calls = []

        def my_callback(trial_info: TrialInfo):
            callback_calls.append(trial_info)

        config = SweepConfig(
            parameters={"x": [0.5, 0.9, 0.3]},  # 0.9 will be best (maximize)
            n_trials=3,
            backend=SweepBackend.GRID_SEARCH,
            direction="maximize",
            post_trial_callback=my_callback,
        )

        def train_fn(params):
            return params["x"]  # Higher is better

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.output_dir = tmp_dir
            sweep = HyperparameterSweep(config, train_fn)
            result = sweep.run()

            assert len(callback_calls) == 3

            # The best one should have the highest metric value
            best_calls = [c for c in callback_calls if c.is_best]
            assert len(best_calls) >= 1
            best_call = best_calls[-1]
            assert best_call.metric_value == max(c.metric_value for c in callback_calls)

    def test_post_trial_script_invoked(self):
        """Test that post_trial_script is executed after each trial."""
        import subprocess

        config = SweepConfig(
            parameters={"x": [0.1, 0.2]},
            n_trials=2,
            backend=SweepBackend.RANDOM_SEARCH,
            post_trial_script="echo 'Trial $TRIAL_NUMBER completed'",
        )

        def train_fn(params):
            return params["x"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.output_dir = tmp_dir

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

                sweep = HyperparameterSweep(config, train_fn)
                result = sweep.run()

                # Script should be called twice (once per trial)
                assert mock_run.call_count == 2

                # Verify script was called with shell=True
                for call in mock_run.call_args_list:
                    assert call[1]["shell"] is True

    def test_post_trial_script_env_vars(self):
        """Test that post_trial_script receives correct environment variables."""
        captured_envs = []

        def capture_env(*args, **kwargs):
            env = kwargs.get("env", {})
            captured_envs.append({
                "TRIAL_NUMBER": env.get("TRIAL_NUMBER"),
                "TRIAL_METRIC_VALUE": env.get("TRIAL_METRIC_VALUE"),
                "TRIAL_IS_BEST": env.get("TRIAL_IS_BEST"),
                "TRIAL_OUTPUT_DIR": env.get("TRIAL_OUTPUT_DIR"),
                "TRIAL_PARAMS": env.get("TRIAL_PARAMS"),
            })
            return Mock(returncode=0, stdout="", stderr="")

        config = SweepConfig(
            parameters={"lr": [0.001, 0.01]},
            n_trials=2,
            backend=SweepBackend.GRID_SEARCH,
            post_trial_script="echo test",
        )

        def train_fn(params):
            return params["lr"] * 100

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.output_dir = tmp_dir

            with patch("subprocess.run", side_effect=capture_env):
                sweep = HyperparameterSweep(config, train_fn)
                result = sweep.run()

            assert len(captured_envs) == 2

            for i, env in enumerate(captured_envs):
                # TRIAL_NUMBER should be set
                assert env["TRIAL_NUMBER"] is not None
                assert int(env["TRIAL_NUMBER"]) >= 0

                # TRIAL_METRIC_VALUE should be set
                assert env["TRIAL_METRIC_VALUE"] is not None
                assert float(env["TRIAL_METRIC_VALUE"]) > 0

                # TRIAL_IS_BEST should be "true" or "false"
                assert env["TRIAL_IS_BEST"] in ("true", "false")

                # TRIAL_PARAMS should contain the parameters
                assert env["TRIAL_PARAMS"] is not None
                assert "lr" in env["TRIAL_PARAMS"]

    def test_post_trial_callback_failure_does_not_stop_sweep(self):
        """Test that callback failure doesn't stop the sweep."""
        call_count = [0]

        def failing_callback(trial_info):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("Simulated callback failure")

        config = SweepConfig(
            parameters={"x": [0.1, 0.2, 0.3]},
            n_trials=3,
            backend=SweepBackend.GRID_SEARCH,
            post_trial_callback=failing_callback,
        )

        def train_fn(params):
            return params["x"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.output_dir = tmp_dir

            with patch("autotrain.utils.logger.warning") as mock_warning:
                sweep = HyperparameterSweep(config, train_fn)
                result = sweep.run()

                # Sweep should complete despite callback failure
                assert len(result.trials) == 3
                # Warning should have been logged
                assert mock_warning.called

    def test_post_trial_script_failure_does_not_stop_sweep(self):
        """Test that script failure doesn't stop the sweep."""
        config = SweepConfig(
            parameters={"x": [0.1, 0.2]},
            n_trials=2,
            backend=SweepBackend.RANDOM_SEARCH,
            post_trial_script="exit 1",  # Script that fails
        )

        def train_fn(params):
            return params["x"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.output_dir = tmp_dir

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = Mock(returncode=1, stdout="", stderr="error")

                with patch("autotrain.utils.logger.warning") as mock_warning:
                    sweep = HyperparameterSweep(config, train_fn)
                    result = sweep.run()

                    # Sweep should complete
                    assert len(result.trials) == 2
                    # Warning should have been logged
                    assert mock_warning.called

    def test_run_autotrain_sweep_with_post_trial_callback(self):
        """Test that run_autotrain_sweep passes post_trial_callback to SweepConfig."""
        from autotrain.utils import TrialInfo

        callback_calls = []

        def my_callback(trial_info: TrialInfo):
            callback_calls.append(trial_info)

        sweep_parameters = {"lr": [0.001, 0.01]}

        def mock_train(params):
            return params["lr"] * 100

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_autotrain_sweep(
                model_config={},
                sweep_parameters=sweep_parameters,
                train_function=mock_train,
                n_trials=2,
                backend="grid",
                output_dir=tmp_dir,
                post_trial_callback=my_callback,
            )

            assert len(callback_calls) == 2
            for info in callback_calls:
                assert isinstance(info, TrialInfo)

    def test_run_autotrain_sweep_with_post_trial_script(self):
        """Test that run_autotrain_sweep passes post_trial_script to SweepConfig."""
        sweep_parameters = {"lr": [0.001, 0.01]}

        def mock_train(params):
            return params["lr"] * 100

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

                result = run_autotrain_sweep(
                    model_config={},
                    sweep_parameters=sweep_parameters,
                    train_function=mock_train,
                    n_trials=2,
                    backend="grid",
                    output_dir=tmp_dir,
                    post_trial_script="echo 'Trial done'",
                )

                # Script should be called twice
                assert mock_run.call_count == 2

    @pytest.mark.skipif(not pytest.importorskip("optuna"), reason="Optuna not installed")
    def test_post_trial_callback_with_optuna(self):
        """Test post_trial_callback works with Optuna backend."""
        from autotrain.utils import TrialInfo

        callback_calls = []

        def my_callback(trial_info: TrialInfo):
            callback_calls.append(trial_info)

        config = SweepConfig(
            parameters={
                "lr": ParameterRange(low=1e-5, high=1e-3, distribution="log_uniform"),
            },
            n_trials=3,
            backend=SweepBackend.OPTUNA,
            post_trial_callback=my_callback,
        )

        def train_fn(params):
            return params["lr"] * 100

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.output_dir = tmp_dir
            sweep = HyperparameterSweep(config, train_fn)
            result = sweep.run()

            assert len(callback_calls) == 3
            for info in callback_calls:
                assert isinstance(info, TrialInfo)
                assert "lr" in info.params
