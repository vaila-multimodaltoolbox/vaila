"""Unit tests for shared cross-platform FFmpeg H.264 encode helpers."""

from pathlib import Path

from vaila.ffmpeg_utils import (
    CPU_VIDEO_ENCODER,
    DEFAULT_NVENC_PRESET,
    NVENC_GPU_INDEX,
    NVENC_VIDEO_ENCODER,
    VIDEOTOOLBOX_VIDEO_ENCODER,
    _binary_basenames,
    _iter_ffmpeg_candidates,
    _nvenc_preset,
    clear_ffmpeg_encoder_cache,
    describe_video_encoder,
    encoders_with_cpu_fallback,
    get_ffmpeg_video_encoding_args,
    is_hardware_video_encoder,
)


def test_h264_nvenc_encoding_args_include_gpu_index_and_ada_defaults():
    args = get_ffmpeg_video_encoding_args(NVENC_VIDEO_ENCODER)
    assert args[0:2] == ["-c:v", NVENC_VIDEO_ENCODER]
    assert "-gpu" in args
    assert args[args.index("-gpu") + 1] == str(NVENC_GPU_INDEX)
    assert "-preset" in args
    assert args[args.index("-preset") + 1] == DEFAULT_NVENC_PRESET
    assert "-cq" in args
    assert args[args.index("-cq") + 1] == "18"
    assert "-spatial-aq" in args
    assert args[args.index("-spatial-aq") + 1] == "1"
    assert "-pix_fmt" in args
    assert args[args.index("-pix_fmt") + 1] == "yuv420p"


def test_videotoolbox_encoding_args():
    args = get_ffmpeg_video_encoding_args(VIDEOTOOLBOX_VIDEO_ENCODER)
    assert args[0:2] == ["-c:v", VIDEOTOOLBOX_VIDEO_ENCODER]
    assert "-b:v" in args
    assert "-allow_sw" in args
    assert "-pix_fmt" in args


def test_libx264_encoding_args_use_crf():
    args = get_ffmpeg_video_encoding_args(CPU_VIDEO_ENCODER)
    assert args == [
        "-c:v",
        CPU_VIDEO_ENCODER,
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
    ]


def test_nvenc_preset_env_override(monkeypatch):
    monkeypatch.setenv("VAILA_NVENC_PRESET", "p6")
    assert _nvenc_preset() == "p6"
    args = get_ffmpeg_video_encoding_args(NVENC_VIDEO_ENCODER)
    assert args[args.index("-preset") + 1] == "p6"
    monkeypatch.setenv("VAILA_NVENC_PRESET", "bogus")
    assert _nvenc_preset() == DEFAULT_NVENC_PRESET


def test_hardware_encoder_helpers():
    assert is_hardware_video_encoder(NVENC_VIDEO_ENCODER)
    assert is_hardware_video_encoder(VIDEOTOOLBOX_VIDEO_ENCODER)
    assert not is_hardware_video_encoder(CPU_VIDEO_ENCODER)
    assert encoders_with_cpu_fallback(NVENC_VIDEO_ENCODER) == [
        NVENC_VIDEO_ENCODER,
        CPU_VIDEO_ENCODER,
    ]
    assert encoders_with_cpu_fallback(VIDEOTOOLBOX_VIDEO_ENCODER) == [
        VIDEOTOOLBOX_VIDEO_ENCODER,
        CPU_VIDEO_ENCODER,
    ]
    assert encoders_with_cpu_fallback(CPU_VIDEO_ENCODER) == [CPU_VIDEO_ENCODER]
    assert "VideoToolbox" in describe_video_encoder(VIDEOTOOLBOX_VIDEO_ENCODER)
    assert "libx264" in describe_video_encoder(CPU_VIDEO_ENCODER)


def test_binary_basenames_are_platform_aware(monkeypatch):
    monkeypatch.setattr("vaila.ffmpeg_utils.sys.platform", "win32")
    assert _binary_basenames("ffmpeg") == ["ffmpeg.exe", "ffmpeg"]
    monkeypatch.setattr("vaila.ffmpeg_utils.sys.platform", "linux")
    assert _binary_basenames("ffmpeg") == ["ffmpeg"]


def test_force_cpu_encoder_via_env(monkeypatch):
    monkeypatch.setenv("VAILA_FFMPEG_ENCODER", "libx264")
    clear_ffmpeg_encoder_cache()
    from vaila.ffmpeg_utils import detect_ffmpeg_video_encoder

    assert detect_ffmpeg_video_encoder() == CPU_VIDEO_ENCODER
    clear_ffmpeg_encoder_cache()


def test_cutvideo_imports_shared_nvenc_helpers():
    from vaila import cutvideo

    assert callable(cutvideo.encoders_with_cpu_fallback)
    assert callable(cutvideo.get_ffmpeg_video_encoding_args)
    assert callable(cutvideo.get_video_encode_ffmpeg_path)
    assert callable(cutvideo.is_hardware_video_encoder)


def test_drawboxe_reuses_shared_nvenc_helpers():
    from vaila import drawboxe, ffmpeg_utils

    assert drawboxe.detect_ffmpeg_video_encoder is ffmpeg_utils.detect_ffmpeg_video_encoder
    assert drawboxe.get_ffmpeg_video_encoding_args is ffmpeg_utils.get_ffmpeg_video_encoding_args
    assert drawboxe.run_ffmpeg_encode_with_fallback is ffmpeg_utils.run_ffmpeg_encode_with_fallback
    assert drawboxe.get_video_encode_ffmpeg_path is ffmpeg_utils.get_video_encode_ffmpeg_path
    assert drawboxe.encoders_with_cpu_fallback is ffmpeg_utils.encoders_with_cpu_fallback


def test_nvenc_candidates_include_system_ffmpeg():
    clear_ffmpeg_encoder_cache()
    candidates = _iter_ffmpeg_candidates()
    assert candidates
    assert any(Path(c).name.startswith("ffmpeg") for c in candidates)
