"""
Tests for apply_chat_template flag and pre-formatted data handling.
=================================================================

Tests that:
1. apply_chat_template=False skips template application
2. Pre-formatted data (already_formatted) still gets completion_mask
3. process_data_with_chat_template respects the flag

Uses REAL tokenizers and datasets - no mocks.
"""

import pytest
from datasets import Dataset
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import Optional

import sys
sys.path.insert(0, "src")

from autotrain.trainers.clm.utils import (
    process_data_with_chat_template,
    add_completion_mask,
    get_response_template,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def qwen_tokenizer():
    """Load real Qwen tokenizer - ChatML format."""
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


@pytest.fixture(scope="module")
def gemma_tokenizer():
    """Load real Gemma tokenizer."""
    return AutoTokenizer.from_pretrained("google/gemma-3-270m-it")


@dataclass
class MockConfig:
    """Mock config object for testing."""
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    text_column: str = "messages"
    chat_template: Optional[str] = "tokenizer"
    trainer: str = "sft"
    valid_split: Optional[str] = None
    apply_chat_template: bool = True
    chat_format: Optional[str] = None


# =============================================================================
# Tests for apply_chat_template flag
# =============================================================================

class TestApplyChatTemplateFlag:
    """Tests for the apply_chat_template boolean flag."""

    def test_apply_chat_template_true_processes_messages(self, qwen_tokenizer):
        """When apply_chat_template=True, messages should be converted to text."""
        config = MockConfig(
            chat_template="tokenizer",
            apply_chat_template=True,
            trainer="sft",
        )

        # Dataset with messages column (not formatted)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        dataset = Dataset.from_dict({"messages": [messages]})

        train_data, valid_data = process_data_with_chat_template(
            config, qwen_tokenizer, dataset, None
        )

        # Should have created 'text' column with formatted chat
        assert "text" in train_data.column_names
        text = train_data[0]["text"]

        # Should contain ChatML markers
        assert "<|im_start|>" in text or "assistant" in text.lower()
        print(f"Formatted text: {repr(text[:200])}")

    def test_apply_chat_template_false_skips_processing(self, qwen_tokenizer):
        """When apply_chat_template=False, messages should NOT be processed."""
        config = MockConfig(
            chat_template="tokenizer",
            apply_chat_template=False,
            trainer="sft",
        )

        # Dataset with messages column (not formatted)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        dataset = Dataset.from_dict({"messages": [messages]})

        train_data, valid_data = process_data_with_chat_template(
            config, qwen_tokenizer, dataset, None
        )

        # Should NOT have created 'text' column (or if it exists, not from processing)
        # The messages column should still be there unchanged
        assert "messages" in train_data.column_names

        # If text exists, it should NOT be formatted (no template tokens)
        if "text" in train_data.column_names:
            text = train_data[0]["text"]
            # Should not have ChatML markers if not processed
            assert "<|im_start|>" not in str(text), "Text should not be formatted when apply_chat_template=False"


class TestPreformattedDataCompletionMask:
    """Tests for completion_mask on pre-formatted data."""

    def test_preformatted_data_gets_completion_mask(self, qwen_tokenizer):
        """Pre-formatted data (with template tokens) should get completion_mask."""
        config = MockConfig(
            chat_template=None,  # No template set (user has pre-formatted data)
            apply_chat_template=True,
            trainer="sft",
            text_column="text",
        )

        # Create pre-formatted text (as if user processed externally)
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."},
        ]
        preformatted_text = qwen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        print(f"Pre-formatted text: {repr(preformatted_text)}")

        # Dataset with pre-formatted text
        dataset = Dataset.from_dict({"text": [preformatted_text]})

        train_data, valid_data = process_data_with_chat_template(
            config, qwen_tokenizer, dataset, None
        )

        # Should have completion_mask even though chat_template=None
        # because data was detected as already_formatted
        assert "completion_mask" in train_data.column_names, (
            "Pre-formatted data should get completion_mask"
        )

        mask = train_data[0]["completion_mask"]
        assert 0 in mask, "Should have prompt tokens (0)"
        assert 1 in mask, "Should have completion tokens (1)"

        print(f"Completion mask length: {len(mask)}, sum: {sum(mask)}")

    def test_preformatted_gemma_data_gets_completion_mask(self, gemma_tokenizer):
        """Pre-formatted Gemma data should get completion_mask."""
        config = MockConfig(
            model="google/gemma-3-270m-it",
            chat_template=None,
            apply_chat_template=True,
            trainer="sft",
            text_column="text",
        )

        # Create pre-formatted Gemma text
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        preformatted_text = gemma_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        print(f"Pre-formatted Gemma text: {repr(preformatted_text)}")

        dataset = Dataset.from_dict({"text": [preformatted_text]})

        train_data, valid_data = process_data_with_chat_template(
            config, gemma_tokenizer, dataset, None
        )

        # Should have completion_mask
        assert "completion_mask" in train_data.column_names
        mask = train_data[0]["completion_mask"]
        assert 1 in mask, "Should have completion tokens"

    def test_apply_chat_template_false_with_preformatted_gets_mask(self, qwen_tokenizer):
        """Even with apply_chat_template=False, pre-formatted data gets completion_mask."""
        config = MockConfig(
            chat_template="tokenizer",  # Template set
            apply_chat_template=False,  # But disabled
            trainer="sft",
            text_column="text",
        )

        # Pre-formatted text (user processed externally)
        messages = [
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "Response"},
        ]
        preformatted_text = qwen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        dataset = Dataset.from_dict({"text": [preformatted_text]})

        train_data, valid_data = process_data_with_chat_template(
            config, qwen_tokenizer, dataset, None
        )

        # Should have completion_mask because data is already_formatted
        assert "completion_mask" in train_data.column_names
        mask = train_data[0]["completion_mask"]
        assert 1 in mask, "Pre-formatted data should get completion_mask even with apply_chat_template=False"


class TestNoTemplateNoMask:
    """Test that unformatted data without chat_template gets no mask."""

    def test_unformatted_data_no_template_no_mask(self, qwen_tokenizer):
        """Unformatted data without chat_template should NOT get completion_mask."""
        config = MockConfig(
            chat_template=None,  # No template
            apply_chat_template=True,
            trainer="sft",
            text_column="text",
        )

        # Plain text (not formatted with any template)
        plain_text = "This is just plain text without any chat formatting."

        dataset = Dataset.from_dict({"text": [plain_text]})

        train_data, valid_data = process_data_with_chat_template(
            config, qwen_tokenizer, dataset, None
        )

        # Should NOT have completion_mask (no template, not pre-formatted)
        if "completion_mask" in train_data.column_names:
            mask = train_data[0]["completion_mask"]
            # If mask exists, it should be all zeros (no completion detected)
            assert all(m == 0 for m in mask), "Unformatted text should have no completion tokens"


class TestInputIdsAddedForTRL:
    """Test that input_ids column is added for TRL compatibility."""

    def test_formatted_data_gets_input_ids(self, qwen_tokenizer):
        """Formatted data should get input_ids column for TRL 0.26+."""
        config = MockConfig(
            chat_template="tokenizer",
            apply_chat_template=True,
            trainer="sft",
        )

        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        dataset = Dataset.from_dict({"messages": [messages]})

        train_data, valid_data = process_data_with_chat_template(
            config, qwen_tokenizer, dataset, None
        )

        # Should have input_ids for TRL
        assert "input_ids" in train_data.column_names
        input_ids = train_data[0]["input_ids"]
        assert len(input_ids) > 0
        assert all(isinstance(i, int) for i in input_ids)

    def test_preformatted_data_gets_input_ids(self, qwen_tokenizer):
        """Pre-formatted data should also get input_ids."""
        config = MockConfig(
            chat_template=None,
            apply_chat_template=True,
            trainer="sft",
            text_column="text",
        )

        # Pre-formatted text
        messages = [
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "Response"},
        ]
        preformatted_text = qwen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        dataset = Dataset.from_dict({"text": [preformatted_text]})

        train_data, valid_data = process_data_with_chat_template(
            config, qwen_tokenizer, dataset, None
        )

        # Should have input_ids
        assert "input_ids" in train_data.column_names


class TestValidationDataHandling:
    """Test that validation data is also processed correctly."""

    def test_validation_data_gets_completion_mask(self, qwen_tokenizer):
        """Validation data should also get completion_mask."""
        config = MockConfig(
            chat_template="tokenizer",
            apply_chat_template=True,
            trainer="sft",
            valid_split="validation",
        )

        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]

        train_dataset = Dataset.from_dict({"messages": [messages]})
        valid_dataset = Dataset.from_dict({"messages": [messages]})

        train_data, valid_data = process_data_with_chat_template(
            config, qwen_tokenizer, train_dataset, valid_dataset
        )

        # Both should have completion_mask
        assert "completion_mask" in train_data.column_names
        assert "completion_mask" in valid_data.column_names

        # Both should have actual completion tokens
        assert 1 in train_data[0]["completion_mask"]
        assert 1 in valid_data[0]["completion_mask"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
