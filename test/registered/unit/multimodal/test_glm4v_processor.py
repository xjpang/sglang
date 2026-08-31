import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

from sglang.srt.configs.glm5_next_processing import (
    Glm5NextImageProcessor,
    Glm5NextProcessor,
    expand_glm5_next_image_token_ids,
    smart_resize,
)
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.multimodal.customized_mm_processor_utils import (
    _CUSTOMIZED_MM_PROCESSOR,
)
from sglang.srt.multimodal.media_artifacts.glm5_next import (
    Glm5NextImagePreprocessArtifact,
)
from sglang.srt.multimodal.media_artifacts import MediaArtifactInput
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultiModalProcessorOutput,
    BaseMultimodalProcessor as SGLangBaseProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.glm4v import (
    Glm4vImageProcessor,
    _collapse_glm5_next_image_tokens,
)
from sglang.srt.utils.hf_transformers import processor as processor_module


def test_glm5_next_image_processor_uses_dynamic_padded_grid():
    pixels = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)

    output = Glm5NextImageProcessor()(
        images=Image.fromarray(pixels), return_tensors="pt"
    )

    assert output.image_grid_thw.tolist() == [[1, 8, 10]]
    assert output.pixel_values.shape == (80, 1176)
    assert output.pixel_values.isfinite().all()


def test_glm5_next_smart_resize_rejects_impossible_token_budget():
    with pytest.raises(ValueError, match="max_pixels=0 is too small"):
        smart_resize(
            num_frames=2,
            height=64,
            width=64,
            temporal_factor=2,
            factor=28,
            max_pixels=0,
        )


def test_glm5_next_expands_image_tokens_without_retokenizing():
    input_ids, attention_mask = expand_glm5_next_image_token_ids(
        input_ids=[[1, 10, 99, 11, 2], [1, 10, 99, 11, 3]],
        attention_mask=[[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]],
        image_token_id=99,
        image_token_counts=[2, 3],
    )

    assert input_ids == [
        [1, 10, 99, 99, 11, 2],
        [1, 10, 99, 99, 99, 11, 3],
    ]
    assert attention_mask == [
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0],
    ]


def test_glm5_next_image_token_expansion_validates_alignment():
    with pytest.raises(ValueError, match="1 image placeholder"):
        expand_glm5_next_image_token_ids([[99]], None, 99, [1, 1])

    with pytest.raises(ValueError, match="more image placeholders"):
        expand_glm5_next_image_token_ids([[99, 99]], None, 99, [1])


def test_glm5_next_processor_uses_direct_image_token_expansion():
    processor = object.__new__(Glm5NextProcessor)
    processor.image_token_id = 99
    processor.image_processor = MagicMock(
        merge_size=2,
        return_value={
            "pixel_values": torch.ones((16, 3)),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        },
    )
    processor.tokenizer = MagicMock(
        init_kwargs={},
        return_value={
            "input_ids": [[1, 10, 99, 11, 2]],
            "attention_mask": [[1, 1, 1, 1, 1]],
        },
    )
    processor._merge_kwargs = MagicMock(
        return_value={
            "images_kwargs": {},
            "text_kwargs": {
                "return_tensors": "pt",
                "return_mm_token_type_ids": True,
            },
        }
    )

    output = processor(images=[Image.new("RGB", (28, 28))], text=["prompt"])

    assert output.input_ids.tolist() == [[1, 10, 99, 99, 99, 99, 11, 2]]
    assert output.attention_mask.tolist() == [[1] * 8]
    assert output.mm_token_type_ids.tolist() == [[0, 0, 1, 1, 1, 1, 0, 0]]
    processor.tokenizer.assert_called_once_with(["prompt"])


def test_collapse_glm5_next_image_tokens_preserves_image_boundaries():
    image_token_id = 99
    image_start_token_id = 10
    image_end_token_id = 11
    input_ids = [10, 99, 99, 11, 10, 99, 99, 99, 11]

    assert _collapse_glm5_next_image_tokens(input_ids, image_token_id) == [
        image_start_token_id,
        99,
        image_end_token_id,
        image_start_token_id,
        99,
        image_end_token_id,
    ]


def test_collapse_glm5_next_image_tokens_handles_sequence_boundaries():
    assert _collapse_glm5_next_image_tokens([99, 99, 1, 99, 99], 99) == [
        99,
        1,
        99,
    ]
    assert _collapse_glm5_next_image_tokens([], 99) == []


def test_glm5_next_enables_isolated_preprocessing_workers():
    hf_config = SimpleNamespace(
        model_type="glm5_next",
        image_token_id=99,
        video_token_id=98,
        image_start_token_id=10,
        image_end_token_id=11,
        video_start_token_id=12,
        video_end_token_id=13,
    )
    hf_processor = SimpleNamespace()

    with patch.object(SGLangBaseProcessor, "__init__", return_value=None):
        processor = Glm4vImageProcessor(hf_config, MagicMock(), hf_processor)

    assert processor.auto_mm_processor_worker_num == 2
    assert processor.auto_mm_io_worker_num == 16
    assert processor.auto_mm_preprocess_cache_size_mb == 4096
    assert processor.supports_mm_processor_concurrency is True


def test_glm5_next_composes_cached_image_artifact_without_hf_processor(monkeypatch):
    image_token_id = 99
    artifact = Glm5NextImagePreprocessArtifact(
        content_digest="sha256:" + "1" * 64,
        artifact_key="sha256:" + "2" * 64,
        feature_hash=1234,
        grid_thw=(1, 4, 4),
        feature=torch.ones((16, 3)),
    )
    processor = object.__new__(Glm4vImageProcessor)
    processor.hf_config = SimpleNamespace(
        image_token_id=image_token_id,
        video_start_token_id=12,
        video_end_token_id=13,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )
    processor.IM_TOKEN_ID = image_token_id
    processor.mm_tokens = MultimodalSpecialTokens(
        image_token_id=image_token_id, video_token_id=98
    )
    processor.use_cuda_ipc = False
    monkeypatch.setattr(
        MRotaryEmbedding,
        "get_rope_index_glm4v",
        MagicMock(
            return_value=(
                torch.zeros((3, 1, 8)),
                torch.zeros(1),
            )
        ),
    )

    output = processor.compose_glm5_next_image_request(
        [1, 10, image_token_id, 11, 2], [artifact]
    )

    assert output.input_ids == [1, 10, 99, 99, 99, 99, 11, 2]
    assert output.mm_items[0].feature is artifact.feature
    assert output.mm_items[0].hash == 1234
    assert output.mm_items[0].offsets == [(2, 5)]
    assert output.mm_items[0].image_grid_thw.tolist() == [[1, 4, 4]]


def test_glm5_next_prepares_individually_cacheable_image_artifacts():
    feature = torch.arange(20 * 3, dtype=torch.float32).reshape(20, 3)
    image_processor = MagicMock(
        return_value={
            "pixel_values": feature,
            "image_grid_thw": torch.tensor([[1, 4, 4], [1, 2, 2]]),
        }
    )
    processor = object.__new__(Glm4vImageProcessor)
    processor._processor = SimpleNamespace(image_processor=image_processor)
    processor.image_config = {"max_image_tokens": 2048}
    processor.disable_fast_image_processor = True
    processor.keep_mm_features_on_device = False
    entries = [
        MediaArtifactInput(
            content_digest="sha256:" + str(index) * 64,
            artifact_key="sha256:" + str(index + 1) * 64,
            modality=Modality.IMAGE,
            media=Image.new("RGB", (28, 28)),
        )
        for index in range(2)
    ]

    artifacts = processor.prepare_artifact_batch(entries)

    image_processor.assert_called_once()
    assert image_processor.call_args.kwargs["max_image_tokens"] == 2048
    assert [artifact.feature.shape for artifact in artifacts] == [(16, 3), (4, 3)]
    assert [artifact.grid_thw for artifact in artifacts] == [(1, 4, 4), (1, 2, 2)]
    assert artifacts[0].feature_hash == int("1" * 16, 16)
    assert artifacts[1].feature_hash == int("2" * 16, 16)


def test_glm5_next_image_only_request_uses_preprocess_cache():
    cached_output = MagicMock()
    artifacts = [MagicMock()]
    processor = object.__new__(Glm4vImageProcessor)
    processor.hf_config = SimpleNamespace(model_type="glm5_next")
    processor.IM_TOKEN_ID = 99
    processor.skip_tokenizer_init = False
    processor.keep_mm_features_on_device = False
    processor.mm_preprocess_cache = SimpleNamespace(enabled=True)
    processor.prepare_media_artifacts = AsyncMock(return_value=artifacts)
    processor.compose_glm5_next_image_request = MagicMock(return_value=cached_output)
    content_hash = "sha256:" + "a" * 64

    output = asyncio.run(
        processor.process_mm_data_async(
            image_data=["image.jpg"],
            input_text="<|begin_of_image|><|image|><|end_of_image|>",
            request_obj=SimpleNamespace(
                video_data=None, mm_content_hashes=[content_hash]
            ),
        )
    )

    assert output is cached_output
    processor.prepare_media_artifacts.assert_awaited_once_with(
        ["image.jpg"], content_hashes=[content_hash]
    )
    processor.compose_glm5_next_image_request.assert_called_once_with(
        "<|begin_of_image|><|image|><|end_of_image|>", artifacts
    )


def test_glm5_next_preserves_processor_expanded_image_tokens(monkeypatch):
    image_token_id = 99
    expanded_input_ids = [1, 10, 99, 99, 99, 99, 11, 2]
    collapsed_input_ids = [1, 10, 99, 11, 2]
    image = Image.new("RGB", (28, 28))
    image_item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=torch.ones((4, 3)),
        model_specific_data={"image_grid_thw": torch.tensor([[1, 2, 2]])},
    )
    processor_output = SimpleNamespace(
        image_grid_thw=torch.tensor([[1, 2, 2]]),
        video_grid_thw=None,
        attention_mask=torch.ones((1, len(expanded_input_ids))),
    )

    processor = object.__new__(Glm4vImageProcessor)
    processor.hf_config = SimpleNamespace(model_type="glm5_next")
    processor.IM_TOKEN_ID = image_token_id
    processor.mm_tokens = MultimodalSpecialTokens(
        image_token_id=image_token_id, video_token_id=98
    )
    processor.video_config = {}
    processor._processor = SimpleNamespace(
        video_processor=None,
        _get_num_multimodal_tokens=lambda **_: SimpleNamespace(num_image_tokens=[4]),
    )
    processor._tokenizer = MagicMock()
    processor.preserve_processor_input_ids = False
    processor.precompute_hash_before_cpu_transfer = False
    processor.use_cuda_ipc = False
    processor.mm_processor_executor = None
    processor.mm_preprocess_cache = SimpleNamespace(enabled=False)
    processor.skip_tokenizer_init = False
    processor.load_mm_data = AsyncMock(
        return_value=BaseMultiModalProcessorOutput(
            input_text="unused", input_ids=collapsed_input_ids, images=[image]
        )
    )
    processor._process_and_collect_mm_items = MagicMock(
        return_value=(
            [image_item],
            torch.tensor(expanded_input_ids),
            processor_output,
        )
    )
    process_and_combine_mm_data_async = AsyncMock(
        wraps=processor.process_and_combine_mm_data_async
    )
    processor.process_and_combine_mm_data_async = process_and_combine_mm_data_async
    monkeypatch.setattr(
        MRotaryEmbedding,
        "get_rope_index_glm4v",
        MagicMock(
            return_value=(
                torch.zeros((3, 1, len(expanded_input_ids))),
                torch.zeros(1),
            )
        ),
    )

    output = asyncio.run(
        processor.process_mm_data_async(
            image_data=["image"],
            input_text=expanded_input_ids,
            request_obj=SimpleNamespace(video_data=None),
        )
    )

    assert processor.load_mm_data.await_args.kwargs["prompt"] == collapsed_input_ids
    process_and_combine_mm_data_async.assert_awaited_once()
    assert output.input_ids == expanded_input_ids
    assert len(output.mm_items) == 1
    assert output.mm_items[0].offsets == [(2, 5)]


def test_get_processor_uses_registered_glm5_next_processor(monkeypatch):
    expected_processor = MagicMock()
    expected_processor.tokenizer.chat_template = "test-template"
    monkeypatch.setattr(
        processor_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: type(
            "Config", (), {"model_type": "glm5_next", "language_model_only": False}
        )(),
    )
    monkeypatch.setattr(
        Glm5NextProcessor,
        "from_pretrained",
        lambda *args, **kwargs: expected_processor,
    )

    assert _CUSTOMIZED_MM_PROCESSOR["glm5_next"] is Glm5NextProcessor
    assert processor_module.get_processor("unused-checkpoint") is expected_processor
