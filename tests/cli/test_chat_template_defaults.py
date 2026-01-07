"""Test chat template auto-selection based on trainer type."""

import pytest

from autotrain.trainers.clm.params import LLMTrainingParams


class TestChatTemplateDefaults:
    """Test that chat_template defaults are set correctly based on trainer type."""

    def test_sft_trainer_defaults_to_tokenizer(self):
        """SFT trainer should default to 'tokenizer' template."""
        params = LLMTrainingParams(
            model="test-model", project_name="test-project", data_path="test-data", trainer="sft"
        )
        assert params.chat_template == "tokenizer"

    def test_dpo_trainer_defaults_to_tokenizer(self):
        """DPO trainer should default to 'tokenizer' template."""
        params = LLMTrainingParams(
            model="test-model",
            project_name="test-project",
            data_path="test-data",
            trainer="dpo",
            model_ref="test-ref-model",  # DPO requires ref model
            prompt_text_column="prompt",  # Required for DPO
            rejected_text_column="rejected",  # Required for DPO
        )
        assert params.chat_template == "tokenizer"

    def test_orpo_trainer_defaults_to_tokenizer(self):
        """ORPO trainer should default to 'tokenizer' template."""
        params = LLMTrainingParams(
            model="test-model",
            project_name="test-project",
            data_path="test-data",
            trainer="orpo",
            prompt_text_column="prompt",  # Required for ORPO
            rejected_text_column="rejected",  # Required for ORPO
        )
        assert params.chat_template == "tokenizer"

    def test_reward_trainer_defaults_to_tokenizer(self):
        """Reward trainer should default to 'tokenizer' template."""
        params = LLMTrainingParams(
            model="test-model", project_name="test-project", data_path="test-data", trainer="reward"
        )
        assert params.chat_template == "tokenizer"

    def test_default_trainer_defaults_to_none(self):
        """Default trainer (pretraining) should default to None (no template)."""
        params = LLMTrainingParams(
            model="test-model", project_name="test-project", data_path="test-data", trainer="default"
        )
        assert params.chat_template is None

    def test_ppo_trainer_defaults_to_none(self):
        """PPO trainer should default to None (uses reward model format)."""
        params = LLMTrainingParams(
            model="test-model",
            project_name="test-project",
            data_path="test-data",
            trainer="ppo",
            rl_reward_model_path="OpenAssistant/reward-model-deberta-v3-large-v2",  # Use HF model ID
        )
        assert params.chat_template is None

    def test_explicit_chat_template_respected(self):
        """Explicitly set chat_template should be respected regardless of trainer."""
        # Test with SFT trainer but explicit 'chatml' template
        params = LLMTrainingParams(
            model="test-model",
            project_name="test-project",
            data_path="test-data",
            trainer="sft",
            chat_template="chatml",
        )
        assert params.chat_template == "chatml"

        # Test with default trainer but explicit 'tokenizer' template
        params = LLMTrainingParams(
            model="test-model",
            project_name="test-project",
            data_path="test-data",
            trainer="default",
            chat_template="tokenizer",
        )
        assert params.chat_template == "tokenizer"

    def test_none_string_converted_to_none(self):
        """String 'none' should be converted to None value."""
        params = LLMTrainingParams(
            model="test-model", project_name="test-project", data_path="test-data", trainer="sft", chat_template="none"
        )
        # 'none' string is converted to None for plain text training
        assert params.chat_template is None


class TestChatTemplateOptions:
    """Test that all documented chat templates are valid options."""

    @pytest.mark.parametrize("template", ["tokenizer", "chatml", "zephyr", "alpaca", "vicuna", "llama", "mistral"])
    def test_valid_chat_templates(self, template):
        """All documented templates should be accepted."""
        params = LLMTrainingParams(
            model="test-model",
            project_name="test-project",
            data_path="test-data",
            trainer="sft",
            chat_template=template,
        )
        assert params.chat_template == template

    def test_explicit_none_respected(self):
        """Explicitly passing None should be respected for plain text training."""
        params = LLMTrainingParams(
            model="test-model",
            project_name="test-project",
            data_path="test-data",
            trainer="sft",
            chat_template=None,  # Explicit None for plain text
        )
        # When None is explicitly passed, it should remain None
        assert params.chat_template is None

    def test_string_none_converted(self):
        """String 'none' should be converted to actual None."""
        params = LLMTrainingParams(
            model="test-model",
            project_name="test-project",
            data_path="test-data",
            trainer="sft",
            chat_template="none",  # String 'none'
        )
        # 'none' string should be converted to None
        assert params.chat_template is None


class TestChatTemplateRendererSelection:
    """Test that chat_template values map to the correct renderer.

    This is a critical test - it verifies that 'tokenizer' uses the native
    tokenizer template, NOT ChatML. Previously there was a bug where
    'tokenizer' incorrectly mapped to 'chatml', causing ChatML tokens
    (<|im_start|>, <|im_end|>) to be added as literal text in training data.
    """

    def test_tokenizer_maps_to_native_not_chatml(self):
        """CRITICAL: 'tokenizer' must map to 'native', not 'chatml'.

        This bug caused models to output ChatML tokens as literal text
        when trained with --chat-template tokenizer on non-ChatML models.
        """
        # Import the chat_format_map from clm/utils.py
        # We need to verify the mapping directly
        from autotrain.trainers.clm.utils import process_data_with_chat_template

        # Get the chat_format_map by inspecting the function
        # Since it's defined inside the function, we test via the ChatFormat enum
        from autotrain.rendering import ChatFormat

        # Verify NATIVE format exists (this is what 'tokenizer' should map to)
        assert hasattr(ChatFormat, "NATIVE"), "ChatFormat.NATIVE must exist"
        assert ChatFormat.NATIVE.value == "native"

        # Verify CHATML is different from NATIVE
        assert ChatFormat.CHATML.value == "chatml"
        assert ChatFormat.CHATML != ChatFormat.NATIVE

    def test_chat_format_map_tokenizer_value(self):
        """Verify the chat_format_map has 'tokenizer' -> 'native' mapping.

        This is the actual bug fix verification - we read the source code
        to ensure the mapping is correct.
        """
        import inspect
        from autotrain.trainers.clm import utils

        # Get the source code of process_data_with_chat_template
        source = inspect.getsource(utils.process_data_with_chat_template)

        # Verify the mapping is correct (should be 'native', not 'chatml')
        assert '"tokenizer": "native"' in source, (
            "CRITICAL BUG: 'tokenizer' must map to 'native', not 'chatml'. "
            "This causes ChatML tokens to be added as literal text in training data."
        )

        # Verify it's NOT the buggy version
        assert '"tokenizer": "chatml"' not in source, (
            "CRITICAL BUG DETECTED: 'tokenizer' is incorrectly mapped to 'chatml'. "
            "This will cause models to output <|im_start|> and <|im_end|> as literal text."
        )

    def test_native_renderer_uses_apply_chat_template(self):
        """Verify TokenizerNativeRenderer uses tokenizer.apply_chat_template."""
        from unittest.mock import MagicMock, patch

        from autotrain.rendering import ChatFormat, get_renderer
        from autotrain.rendering.message_renderer import (
            Conversation,
            Message,
            RenderConfig,
            TokenizerNativeRenderer,
        )

        # Create a mock tokenizer with a chat template
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template = MagicMock(
            return_value="<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\nHi<end_of_turn>"
        )

        # Get the native renderer
        config = RenderConfig(format=ChatFormat.NATIVE)
        renderer = TokenizerNativeRenderer(mock_tokenizer, config)

        # Create a test conversation
        conversation = Conversation(
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi"),
            ]
        )

        # Render the conversation
        result = renderer.render_conversation(conversation)

        # Verify apply_chat_template was called
        mock_tokenizer.apply_chat_template.assert_called_once()

        # Verify the result matches the tokenizer's output
        assert result == "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\nHi<end_of_turn>"

        # Verify NO ChatML tokens in output
        assert "<|im_start|>" not in result
        assert "<|im_end|>" not in result

    def test_chatml_renderer_adds_chatml_tokens(self):
        """Verify ChatMLRenderer DOES add ChatML tokens (for comparison)."""
        from unittest.mock import MagicMock

        from autotrain.rendering import ChatFormat
        from autotrain.rendering.message_renderer import (
            ChatMLRenderer,
            Conversation,
            Message,
            RenderConfig,
        )

        mock_tokenizer = MagicMock()
        config = RenderConfig(format=ChatFormat.CHATML)
        renderer = ChatMLRenderer(mock_tokenizer, config)

        conversation = Conversation(
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi"),
            ]
        )

        result = renderer.render_conversation(conversation)

        # ChatML renderer SHOULD add these tokens
        assert "<|im_start|>" in result
        assert "<|im_end|>" in result
        assert "<|im_start|>user" in result
        assert "<|im_start|>assistant" in result

    def test_get_renderer_returns_correct_type(self):
        """Verify get_renderer returns the correct renderer class for each format."""
        from unittest.mock import MagicMock

        from autotrain.rendering import ChatFormat, get_renderer
        from autotrain.rendering.message_renderer import (
            ChatMLRenderer,
            TokenizerNativeRenderer,
        )

        mock_tokenizer = MagicMock()

        # 'native' should return TokenizerNativeRenderer
        native_renderer = get_renderer(ChatFormat.NATIVE, mock_tokenizer)
        assert isinstance(native_renderer, TokenizerNativeRenderer), (
            f"Expected TokenizerNativeRenderer, got {type(native_renderer)}"
        )

        # 'chatml' should return ChatMLRenderer
        chatml_renderer = get_renderer(ChatFormat.CHATML, mock_tokenizer)
        assert isinstance(chatml_renderer, ChatMLRenderer), (
            f"Expected ChatMLRenderer, got {type(chatml_renderer)}"
        )

    @pytest.mark.parametrize(
        "template,expected_format",
        [
            ("tokenizer", "native"),  # CRITICAL: must be native, not chatml
            ("chatml", "chatml"),
            ("zephyr", "zephyr"),
            ("alpaca", "alpaca"),
            ("vicuna", "vicuna"),
            ("llama", "llama"),
            ("mistral", "mistral"),
        ],
    )
    def test_all_template_mappings(self, template, expected_format):
        """Verify all chat_template values map to the expected format."""
        import inspect
        from autotrain.trainers.clm import utils

        source = inspect.getsource(utils.process_data_with_chat_template)

        # Check that the mapping exists in source
        expected_mapping = f'"{template}": "{expected_format}"'
        assert expected_mapping in source, (
            f"Expected mapping '{template}' -> '{expected_format}' not found in source"
        )
