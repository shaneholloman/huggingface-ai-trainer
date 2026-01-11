"""
Tests for Response-Only Training (SFT Label Masking)
=====================================================

This module tests the response-only training feature which ensures the model
only learns to predict assistant/model responses, not prompts/instructions.

Uses REAL tokenizers from HuggingFace - no mocks.

Models tested:
- google/gemma-3-270m-it: Gemma format, NO generation tags (uses completion_mask fallback)
- Qwen/Qwen2.5-0.5B-Instruct: ChatML format, NO generation tags (uses completion_mask fallback)
- HuggingFaceTB/SmolLM3-3B: HAS generation tags (uses TRL native assistant_only_loss)
"""

import pytest
from datasets import Dataset
from transformers import AutoTokenizer

import sys
sys.path.insert(0, "src")

from autotrain.trainers.clm.utils import (
    has_generation_tags,
    get_response_template,
    add_completion_mask,
    get_model_turn_markers,
)


# =============================================================================
# Fixtures - Load REAL tokenizers
# =============================================================================

@pytest.fixture(scope="module")
def gemma_tokenizer():
    """Load real Gemma tokenizer - NO generation tags."""
    return AutoTokenizer.from_pretrained("google/gemma-3-270m-it")


@pytest.fixture(scope="module")
def qwen_tokenizer():
    """Load real Qwen tokenizer - NO generation tags, ChatML format."""
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


@pytest.fixture(scope="module")
def smollm_tokenizer():
    """Load real SmolLM3 tokenizer - HAS generation tags."""
    return AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")


# =============================================================================
# Tests for has_generation_tags()
# =============================================================================

class TestHasGenerationTags:
    """Tests for detecting {% generation %} tags in chat templates."""

    def test_gemma_no_generation_tags(self, gemma_tokenizer):
        """Gemma should NOT have generation tags."""
        assert has_generation_tags(gemma_tokenizer) is False

    def test_qwen_no_generation_tags(self, qwen_tokenizer):
        """Qwen (ChatML) should NOT have generation tags."""
        assert has_generation_tags(qwen_tokenizer) is False

    def test_smollm_has_generation_tags(self, smollm_tokenizer):
        """SmolLM3 SHOULD have generation tags."""
        assert has_generation_tags(smollm_tokenizer) is True


# =============================================================================
# Tests for get_response_template()
# =============================================================================

class TestGetResponseTemplate:
    """Tests for detecting response templates for various model formats."""

    def test_gemma_response_template(self, gemma_tokenizer):
        """Should detect Gemma's response template: <start_of_turn>model\\n"""
        template = get_response_template(gemma_tokenizer)
        assert template is not None
        assert template == "<start_of_turn>model\n"

    def test_qwen_response_template(self, qwen_tokenizer):
        """Should detect Qwen's ChatML response template: <|im_start|>assistant\\n"""
        template = get_response_template(qwen_tokenizer)
        assert template is not None
        assert "<|im_start|>assistant" in template

    def test_smollm_response_template(self, smollm_tokenizer):
        """SmolLM3 should also have a detectable response template."""
        template = get_response_template(smollm_tokenizer)
        # SmolLM uses ChatML-like format
        assert template is not None


# =============================================================================
# Tests for add_completion_mask() - REAL tokenizers
# =============================================================================

class TestAddCompletionMaskGemma:
    """Tests for add_completion_mask with real Gemma tokenizer."""

    def test_single_turn_gemma(self, gemma_tokenizer):
        """Test single turn conversation with Gemma."""
        # Create text using Gemma's chat template
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
        ]
        text = gemma_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        print(f"\n=== Gemma Single Turn ===")
        print(f"Text: {repr(text)}")

        dataset = Dataset.from_dict({"text": [text]})
        response_template = get_response_template(gemma_tokenizer)

        result = add_completion_mask(dataset, gemma_tokenizer, response_template, "text")

        mask = result[0]["completion_mask"]
        tokens = gemma_tokenizer.encode(text, add_special_tokens=False)

        # Verify mask length matches tokens
        assert len(mask) == len(tokens), f"Mask length {len(mask)} != tokens length {len(tokens)}"

        # Verify we have both prompt (0) and completion (1) tokens
        assert 0 in mask, "Should have prompt tokens (0)"
        assert 1 in mask, "Should have completion tokens (1)"

        # Calculate ratio
        completion_ratio = sum(mask) / len(mask)
        print(f"Completion ratio: {completion_ratio:.1%}")

        # In this case, ~half should be completion (assistant response)
        assert 0.2 < completion_ratio < 0.8, f"Unexpected completion ratio: {completion_ratio}"

        # Debug: show which tokens are marked as completion
        print("\nToken analysis:")
        for i, (tok_id, m) in enumerate(zip(tokens, mask)):
            tok_text = gemma_tokenizer.decode([tok_id])
            marker = "COMPLETION" if m == 1 else "prompt"
            print(f"  {i}: {repr(tok_text):20} -> {marker}")

    def test_multi_turn_gemma(self, gemma_tokenizer):
        """Test multi-turn conversation with Gemma."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
            {"role": "user", "content": "And 3+3?"},
            {"role": "assistant", "content": "3+3 equals 6."},
        ]
        text = gemma_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        print(f"\n=== Gemma Multi Turn ===")
        print(f"Text: {repr(text)}")

        dataset = Dataset.from_dict({"text": [text]})
        response_template = get_response_template(gemma_tokenizer)

        result = add_completion_mask(dataset, gemma_tokenizer, response_template, "text")

        mask = result[0]["completion_mask"]
        tokens = gemma_tokenizer.encode(text, add_special_tokens=False)

        assert len(mask) == len(tokens)
        assert 0 in mask, "Should have prompt tokens"
        assert 1 in mask, "Should have completion tokens"

        completion_ratio = sum(mask) / len(mask)
        print(f"Completion ratio: {completion_ratio:.1%}")

        # Multi-turn: both assistant responses should be marked
        # So completion should be roughly similar portion
        assert 0.2 < completion_ratio < 0.7, f"Unexpected completion ratio: {completion_ratio}"

    def test_with_system_prompt_gemma(self, gemma_tokenizer):
        """Test conversation with system prompt - system should NOT be completion."""
        messages = [
            {"role": "user", "content": "You are a helpful assistant.\n\nHello!"},
            {"role": "assistant", "content": "Hello! How can I help?"},
        ]
        text = gemma_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        print(f"\n=== Gemma With System ===")

        dataset = Dataset.from_dict({"text": [text]})
        response_template = get_response_template(gemma_tokenizer)

        result = add_completion_mask(dataset, gemma_tokenizer, response_template, "text")

        mask = result[0]["completion_mask"]
        tokens = gemma_tokenizer.encode(text, add_special_tokens=False)

        # System prompt should be in the non-completion portion
        assert 0 in mask, "System/user tokens should be masked"
        assert 1 in mask, "Assistant tokens should be completion"


class TestAddCompletionMaskQwen:
    """Tests for add_completion_mask with real Qwen (ChatML) tokenizer."""

    def test_single_turn_qwen(self, qwen_tokenizer):
        """Test single turn conversation with Qwen."""
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        text = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        print(f"\n=== Qwen Single Turn ===")
        print(f"Text: {repr(text)}")

        dataset = Dataset.from_dict({"text": [text]})
        response_template = get_response_template(qwen_tokenizer)

        result = add_completion_mask(dataset, qwen_tokenizer, response_template, "text")

        mask = result[0]["completion_mask"]
        tokens = qwen_tokenizer.encode(text, add_special_tokens=False)

        assert len(mask) == len(tokens)
        assert 0 in mask, "Should have prompt tokens"
        assert 1 in mask, "Should have completion tokens"

        completion_ratio = sum(mask) / len(mask)
        print(f"Completion ratio: {completion_ratio:.1%}")

    def test_multi_turn_qwen(self, qwen_tokenizer):
        """Test multi-turn conversation with Qwen."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Bye"},
            {"role": "assistant", "content": "Goodbye!"},
        ]
        text = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        print(f"\n=== Qwen Multi Turn ===")

        dataset = Dataset.from_dict({"text": [text]})
        response_template = get_response_template(qwen_tokenizer)

        result = add_completion_mask(dataset, qwen_tokenizer, response_template, "text")

        mask = result[0]["completion_mask"]

        assert 0 in mask, "Should have prompt/system tokens"
        assert 1 in mask, "Should have completion tokens"


# =============================================================================
# Tests for completion mask correctness
# =============================================================================

class TestCompletionMaskCorrectness:
    """Verify the mask correctly identifies assistant vs non-assistant tokens."""

    def test_end_of_turn_included_in_completion(self, gemma_tokenizer):
        """The end-of-turn token should be INCLUDED in completion (so model learns to stop)."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        text = gemma_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        dataset = Dataset.from_dict({"text": [text]})
        response_template = get_response_template(gemma_tokenizer)
        result = add_completion_mask(dataset, gemma_tokenizer, response_template, "text")

        mask = result[0]["completion_mask"]
        tokens = gemma_tokenizer.encode(text, add_special_tokens=False)

        # Find the <end_of_turn> token ID after assistant response
        end_turn_id = gemma_tokenizer.encode("<end_of_turn>", add_special_tokens=False)[0]

        # Find positions where end_of_turn appears
        end_turn_positions = [i for i, t in enumerate(tokens) if t == end_turn_id]

        print(f"\nEnd turn positions: {end_turn_positions}")
        print(f"Mask at those positions: {[mask[i] for i in end_turn_positions]}")

        # At least the last end_of_turn (after assistant) should be marked as completion
        # The first one (after user) should not be
        if len(end_turn_positions) >= 2:
            # First end_turn is after user - should be 0
            assert mask[end_turn_positions[0]] == 0, "End turn after user should NOT be completion"
            # Second end_turn is after assistant - should be 1
            assert mask[end_turn_positions[1]] == 1, "End turn after assistant SHOULD be completion"

    def test_user_turn_not_in_completion(self, gemma_tokenizer):
        """User messages should NOT be marked as completion."""
        messages = [
            {"role": "user", "content": "UNIQUE_USER_TEXT"},
            {"role": "assistant", "content": "response"},
        ]
        text = gemma_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        dataset = Dataset.from_dict({"text": [text]})
        response_template = get_response_template(gemma_tokenizer)
        result = add_completion_mask(dataset, gemma_tokenizer, response_template, "text")

        mask = result[0]["completion_mask"]
        tokens = gemma_tokenizer.encode(text, add_special_tokens=False)

        # Tokenize the user text to find its tokens
        user_tokens = gemma_tokenizer.encode("UNIQUE_USER_TEXT", add_special_tokens=False)

        # Find where user tokens appear in the full sequence
        for i in range(len(tokens) - len(user_tokens) + 1):
            if tokens[i:i+len(user_tokens)] == user_tokens:
                # All these positions should be 0 (not completion)
                user_mask = mask[i:i+len(user_tokens)]
                assert all(m == 0 for m in user_mask), f"User tokens should not be completion, got mask: {user_mask}"
                break


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_empty_text(self, gemma_tokenizer):
        """Should handle empty text gracefully."""
        dataset = Dataset.from_dict({"text": [""]})
        response_template = get_response_template(gemma_tokenizer)
        result = add_completion_mask(dataset, gemma_tokenizer, response_template, "text")

        assert "completion_mask" in result.column_names
        assert result[0]["completion_mask"] == []

    def test_no_assistant_response(self, gemma_tokenizer):
        """Should handle text with no assistant response (all 0s)."""
        # Just user message, no assistant
        text = "<bos><start_of_turn>user\nHello<end_of_turn>\n"

        dataset = Dataset.from_dict({"text": [text]})
        response_template = get_response_template(gemma_tokenizer)
        result = add_completion_mask(dataset, gemma_tokenizer, response_template, "text")

        mask = result[0]["completion_mask"]
        # All should be 0 since no assistant response
        assert all(m == 0 for m in mask), "No assistant = all zeros"

    def test_preserves_other_columns(self, gemma_tokenizer):
        """Should preserve other columns in dataset."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        text = gemma_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        dataset = Dataset.from_dict({
            "text": [text],
            "id": [123],
            "metadata": ["test"],
        })
        response_template = get_response_template(gemma_tokenizer)
        result = add_completion_mask(dataset, gemma_tokenizer, response_template, "text")

        assert result[0]["id"] == 123
        assert result[0]["metadata"] == "test"

    def test_batch_of_examples(self, gemma_tokenizer):
        """Should work with multiple examples."""
        messages1 = [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}]
        messages2 = [{"role": "user", "content": "C"}, {"role": "assistant", "content": "D"}]

        text1 = gemma_tokenizer.apply_chat_template(messages1, tokenize=False, add_generation_prompt=False)
        text2 = gemma_tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=False)

        dataset = Dataset.from_dict({"text": [text1, text2]})
        response_template = get_response_template(gemma_tokenizer)
        result = add_completion_mask(dataset, gemma_tokenizer, response_template, "text")

        assert len(result) == 2
        assert all("completion_mask" in result[i] for i in range(2))
        assert all(1 in result[i]["completion_mask"] for i in range(2))


# =============================================================================
# Config Parameter Tests
# =============================================================================

class TestResponseOnlyLossConfig:
    """Tests for the response_only_loss config parameter."""

    def test_default_value(self):
        """response_only_loss should default to True."""
        from autotrain.trainers.clm.params import LLMTrainingParams

        params = LLMTrainingParams(
            model="test/model",
            data_path="test/data",
        )
        assert params.response_only_loss is True

    def test_can_be_disabled(self):
        """response_only_loss can be set to False."""
        from autotrain.trainers.clm.params import LLMTrainingParams

        params = LLMTrainingParams(
            model="test/model",
            data_path="test/data",
            response_only_loss=False,
        )
        assert params.response_only_loss is False


# =============================================================================
# Integration test - verify TRL will use the mask
# =============================================================================

class TestTRLIntegration:
    """Test that our completion_mask works with TRL's DataCollator."""

    def test_completion_mask_format_for_trl(self, gemma_tokenizer):
        """Verify completion_mask is in the format TRL expects."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        text = gemma_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        dataset = Dataset.from_dict({"text": [text]})
        response_template = get_response_template(gemma_tokenizer)
        result = add_completion_mask(dataset, gemma_tokenizer, response_template, "text")

        mask = result[0]["completion_mask"]

        # TRL expects completion_mask to be a list of 0s and 1s
        assert isinstance(mask, list), "completion_mask should be a list"
        assert all(m in [0, 1] for m in mask), "completion_mask should only contain 0s and 1s"

        # Length should match tokenized text
        tokens = gemma_tokenizer.encode(text, add_special_tokens=False)
        assert len(mask) == len(tokens), "completion_mask length should match token count"

    def test_trl_data_collator_uses_completion_mask(self, gemma_tokenizer):
        """Verify TRL's DataCollatorForLanguageModeling actually uses completion_mask to set labels=-100."""
        from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

        # Create conversation
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."},
        ]
        text = gemma_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        # Generate completion_mask using our function
        dataset = Dataset.from_dict({"text": [text]})
        response_template = get_response_template(gemma_tokenizer)
        result = add_completion_mask(dataset, gemma_tokenizer, response_template, "text")

        tokens = gemma_tokenizer.encode(text, add_special_tokens=False)
        mask = result[0]["completion_mask"]

        print(f"\nText: {repr(text)}")
        print(f"Tokens: {len(tokens)}, Mask sum: {sum(mask)}")

        # Create TRL's data collator
        collator = DataCollatorForLanguageModeling(
            pad_token_id=gemma_tokenizer.pad_token_id or gemma_tokenizer.eos_token_id,
            completion_only_loss=True,  # This tells TRL to use completion_mask
        )

        # Collate a batch with completion_mask
        batch = collator([{"input_ids": tokens, "completion_mask": mask}])

        labels = batch["labels"][0].tolist()

        # Count masked vs trained tokens
        masked_count = sum(1 for l in labels if l == -100)
        trained_count = sum(1 for l in labels if l != -100)

        print(f"Labels: masked={masked_count}, trained={trained_count}")

        # Verify: number of trained tokens should equal completion_mask sum
        assert trained_count == sum(mask), (
            f"Mismatch! TRL trained on {trained_count} tokens but completion_mask marked {sum(mask)}"
        )

        # Verify token-by-token: mask=0 -> label=-100, mask=1 -> label=token_id
        for i, (label, m) in enumerate(zip(labels, mask)):
            if m == 0:
                assert label == -100, f"Token {i}: mask=0 but label={label}, expected -100"
            else:
                assert label != -100, f"Token {i}: mask=1 but label=-100, expected actual token"

        print("✓ TRL correctly uses completion_mask!")

    def test_trl_data_collator_masks_user_tokens(self, gemma_tokenizer):
        """Verify user message tokens have label=-100 (not trained on)."""
        from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

        # Use unique text to identify user tokens
        messages = [
            {"role": "user", "content": "USERTOKEN123"},
            {"role": "assistant", "content": "response"},
        ]
        text = gemma_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        dataset = Dataset.from_dict({"text": [text]})
        response_template = get_response_template(gemma_tokenizer)
        result = add_completion_mask(dataset, gemma_tokenizer, response_template, "text")

        tokens = gemma_tokenizer.encode(text, add_special_tokens=False)
        mask = result[0]["completion_mask"]

        collator = DataCollatorForLanguageModeling(
            pad_token_id=gemma_tokenizer.pad_token_id or gemma_tokenizer.eos_token_id,
            completion_only_loss=True,
        )
        batch = collator([{"input_ids": tokens, "completion_mask": mask}])
        labels = batch["labels"][0].tolist()

        # Find where "USERTOKEN123" tokens are
        user_token_ids = gemma_tokenizer.encode("USERTOKEN123", add_special_tokens=False)

        # Find position in full sequence
        for i in range(len(tokens) - len(user_token_ids) + 1):
            if tokens[i:i+len(user_token_ids)] == user_token_ids:
                # All these labels should be -100
                user_labels = labels[i:i+len(user_token_ids)]
                assert all(l == -100 for l in user_labels), (
                    f"User tokens should have label=-100, got: {user_labels}"
                )
                print(f"✓ User tokens at positions {i}-{i+len(user_token_ids)-1} correctly masked")
                return

        pytest.fail("Could not find user tokens in sequence")

    def test_trl_data_collator_trains_on_assistant_tokens(self, gemma_tokenizer):
        """Verify assistant message tokens have actual labels (trained on)."""
        from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

        # Use unique text to identify assistant tokens
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "ASSISTANTTOKEN456"},
        ]
        text = gemma_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        dataset = Dataset.from_dict({"text": [text]})
        response_template = get_response_template(gemma_tokenizer)
        result = add_completion_mask(dataset, gemma_tokenizer, response_template, "text")

        tokens = gemma_tokenizer.encode(text, add_special_tokens=False)
        mask = result[0]["completion_mask"]

        collator = DataCollatorForLanguageModeling(
            pad_token_id=gemma_tokenizer.pad_token_id or gemma_tokenizer.eos_token_id,
            completion_only_loss=True,
        )
        batch = collator([{"input_ids": tokens, "completion_mask": mask}])
        labels = batch["labels"][0].tolist()

        # Find where "ASSISTANTTOKEN456" tokens are
        assistant_token_ids = gemma_tokenizer.encode("ASSISTANTTOKEN456", add_special_tokens=False)

        # Find position in full sequence
        for i in range(len(tokens) - len(assistant_token_ids) + 1):
            if tokens[i:i+len(assistant_token_ids)] == assistant_token_ids:
                # All these labels should NOT be -100
                assistant_labels = labels[i:i+len(assistant_token_ids)]
                assert all(l != -100 for l in assistant_labels), (
                    f"Assistant tokens should have actual labels, got: {assistant_labels}"
                )
                # Labels should match input_ids (shifted by 1 in causal LM, but TRL handles this)
                print(f"✓ Assistant tokens at positions {i}-{i+len(assistant_token_ids)-1} correctly have labels")
                return

        pytest.fail("Could not find assistant tokens in sequence")


# =============================================================================
# End-to-end training test (lightweight)
# =============================================================================

class TestEndToEndTraining:
    """Test actual training runs with response_only_loss."""

    def test_sft_trainer_with_completion_mask(self, gemma_tokenizer):
        """Test that SFTTrainer can run with completion_mask in dataset."""
        import torch
        from transformers import AutoModelForCausalLM
        from trl import SFTTrainer, SFTConfig
        import tempfile
        import os

        # Skip if no GPU (training is slow on CPU)
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"\nRunning on device: {device}")

        # Create small dataset
        messages_list = [
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
            [{"role": "user", "content": "Bye"}, {"role": "assistant", "content": "Goodbye!"}],
        ]

        texts = [
            gemma_tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in messages_list
        ]

        dataset = Dataset.from_dict({"text": texts})

        # Add completion_mask
        response_template = get_response_template(gemma_tokenizer)
        dataset = add_completion_mask(dataset, gemma_tokenizer, response_template, "text")

        # Tokenize
        def tokenize(example):
            tokens = gemma_tokenizer.encode(example["text"], add_special_tokens=False)
            return {"input_ids": tokens, "completion_mask": example["completion_mask"]}

        dataset = dataset.map(tokenize)

        # Verify completion_mask is in dataset
        assert "completion_mask" in dataset.column_names
        assert all(1 in example["completion_mask"] for example in dataset)

        print(f"Dataset size: {len(dataset)}")
        print(f"Sample completion_mask ratio: {sum(dataset[0]['completion_mask'])/len(dataset[0]['completion_mask']):.1%}")

        # Load tiny model for testing
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-3-270m-it",
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device if device != "cpu" else None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Configure trainer for minimal training
            config = SFTConfig(
                output_dir=tmpdir,
                max_steps=2,  # Just 2 steps to verify it works
                per_device_train_batch_size=1,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                completion_only_loss=True,  # Use completion_mask
            )

            trainer = SFTTrainer(
                model=model,
                args=config,
                train_dataset=dataset,
                processing_class=gemma_tokenizer,
            )

            print("Starting training...")
            result = trainer.train()

            print(f"Training completed! Loss: {result.training_loss:.4f}")

            # Verify training actually happened
            assert result.training_loss is not None
            assert result.training_loss > 0
            assert result.global_step == 2

        print("✓ SFTTrainer works with completion_mask!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
