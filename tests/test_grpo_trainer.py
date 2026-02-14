import sys
import tempfile

import pytest
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from autotrain.trainers.clm.params import LLMTrainingParams

TINY_MODEL = "sshleifer/tiny-gpt2"


class SimpleMatchEnv:
    """Minimal env for testing: scores 1.0 if completion contains 'hello', else 0.0."""

    def build_dataset(self, tokenizer):
        prompts = [
            "Say hello to the world",
            "Greet everyone with hello",
            "Write a hello message",
            "Say hi to people",
        ]
        return Dataset.from_dict({
            "prompt": prompts,
            "case_idx": list(range(len(prompts))),
        })

    def score_episode(self, model, tokenizer, completion, case_idx):
        text = completion if isinstance(completion, str) else str(completion)
        return 1.0 if "hello" in text.lower() else 0.0

    def get_tools(self):
        return []


class TestGRPOEnvInterface:
    def test_build_dataset(self):
        env = SimpleMatchEnv()
        tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
        ds = env.build_dataset(tokenizer)
        assert "prompt" in ds.column_names
        assert "case_idx" in ds.column_names
        assert len(ds) == 4

    def test_score_episode(self):
        env = SimpleMatchEnv()
        assert env.score_episode(None, None, "hello world", 0) == 1.0
        assert env.score_episode(None, None, "goodbye", 0) == 0.0

    def test_get_tools(self):
        env = SimpleMatchEnv()
        assert env.get_tools() == []


class TestGRPORewardFnWrapping:
    def test_reward_fn_returns_list_float(self):
        env = SimpleMatchEnv()

        def reward_fn(completions, prompts, **kwargs):
            rewards = []
            case_indices = kwargs.get("case_idx", list(range(len(completions))))
            for i, completion in enumerate(completions):
                case_idx = case_indices[i] if i < len(case_indices) else i
                score = env.score_episode(None, None, completion, case_idx)
                rewards.append(float(score))
            return rewards

        result = reward_fn(
            completions=["hello world", "goodbye", "say hello"],
            prompts=["p1", "p2", "p3"],
            case_idx=[0, 1, 2],
        )
        assert result == [1.0, 0.0, 1.0]
        assert all(isinstance(r, float) for r in result)


class TestGRPOConfigCreation:
    def test_grpo_config_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grpo_config = GRPOConfig(
                output_dir=tmpdir,
                num_generations=2,
                max_completion_length=32,
                temperature=1.0,
                beta=0.0,
                epsilon=0.2,
                loss_type="grpo",
                per_device_train_batch_size=2,
                num_train_epochs=1,
                logging_steps=1,
                report_to="none",
            )
            assert grpo_config.num_generations == 2
            assert grpo_config.loss_type == "grpo"
            assert grpo_config.beta == 0.0


class TestGRPOTrainerRuns:
    @pytest.mark.slow
    def test_grpo_trainer_completes(self):
        """Full integration test: load model, create trainer, run 1 step."""
        tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)

        env = SimpleMatchEnv()
        train_dataset = env.build_dataset(tokenizer)

        def reward_fn(completions, prompts, **kwargs):
            rewards = []
            for completion in completions:
                text = completion if isinstance(completion, str) else str(completion)
                rewards.append(1.0 if "hello" in text.lower() else 0.0)
            return rewards

        with tempfile.TemporaryDirectory() as tmpdir:
            grpo_config = GRPOConfig(
                output_dir=tmpdir,
                num_generations=2,
                max_completion_length=16,
                temperature=1.0,
                beta=0.0,
                epsilon=0.2,
                loss_type="grpo",
                per_device_train_batch_size=2,
                num_train_epochs=1,
                max_steps=1,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                gradient_accumulation_steps=1,
            )

            trainer = GRPOTrainer(
                model=model,
                reward_funcs=reward_fn,
                args=grpo_config,
                train_dataset=train_dataset,
                processing_class=tokenizer,
            )

            trainer.train()


class TestGRPODispatch:
    def test_dispatch_routes_to_grpo(self):
        """Verify __main__.train dispatches grpo correctly."""
        from unittest.mock import patch

        with patch("autotrain.trainers.clm.train_clm_grpo.train") as mock_train:
            from autotrain.trainers.clm.__main__ import train

            config = LLMTrainingParams(
                model=TINY_MODEL,
                trainer="grpo",
                rl_env_module="tests.test_grpo_trainer",
                rl_env_class="SimpleMatchEnv",
            )
            train(config)
            mock_train.assert_called_once_with(config)


class TestGRPOParamsValidation:
    def test_grpo_with_env_params_passes(self):
        config = LLMTrainingParams(
            model=TINY_MODEL,
            trainer="grpo",
            rl_env_module="my_envs.hotel_env",
            rl_env_class="HotelEnv",
        )
        assert config.rl_env_module == "my_envs.hotel_env"
        assert config.rl_env_class == "HotelEnv"
        assert config.rl_num_generations == 4

    def test_grpo_without_env_module_raises(self):
        with pytest.raises(ValueError, match="rl-env-module"):
            LLMTrainingParams(
                model=TINY_MODEL,
                trainer="grpo",
                rl_env_class="HotelEnv",
            )

    def test_grpo_without_env_class_raises(self):
        with pytest.raises(ValueError, match="rl-env-class"):
            LLMTrainingParams(
                model=TINY_MODEL,
                trainer="grpo",
                rl_env_module="my_envs.hotel_env",
            )

    def test_grpo_shares_rl_params(self):
        """Shared RL params should not warn when used with grpo."""
        config = LLMTrainingParams(
            model=TINY_MODEL,
            trainer="grpo",
            rl_env_module="my_envs.hotel_env",
            rl_env_class="HotelEnv",
            rl_temperature=0.7,
            rl_max_new_tokens=256,
            rl_clip_range=0.1,
        )
        assert config.rl_temperature == 0.7
        assert config.rl_max_new_tokens == 256
        assert config.rl_clip_range == 0.1


class TestGRPOFieldScopes:
    def test_grpo_in_valid_trainers(self):
        from autotrain.cli.run_llm import VALID_TRAINERS

        assert "grpo" in VALID_TRAINERS

    def test_grpo_only_params_scoped(self):
        from autotrain.cli.run_llm import FIELD_SCOPES

        assert FIELD_SCOPES["rl_env_module"] == ["grpo"]
        assert FIELD_SCOPES["rl_env_class"] == ["grpo"]
        assert FIELD_SCOPES["rl_num_generations"] == ["grpo"]

    def test_shared_params_include_grpo(self):
        from autotrain.cli.run_llm import FIELD_SCOPES

        shared_params = [
            "rl_temperature", "rl_max_new_tokens", "rl_top_k",
            "rl_top_p", "rl_clip_range", "rl_kl_coef", "rl_env_config",
        ]
        for param in shared_params:
            assert "grpo" in FIELD_SCOPES[param], f"{param} should include grpo in scope"
            assert "ppo" in FIELD_SCOPES[param], f"{param} should still include ppo"

    def test_ppo_only_params_exclude_grpo(self):
        from autotrain.cli.run_llm import FIELD_SCOPES

        ppo_only = [
            "rl_gamma", "rl_gae_lambda", "rl_value_loss_coef",
            "rl_reward_fn", "rl_reward_model_path", "rl_num_ppo_epochs",
        ]
        for param in ppo_only:
            assert "grpo" not in FIELD_SCOPES[param], f"{param} should NOT include grpo"
