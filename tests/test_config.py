"""
Configuration validation tests (plan §34.2).

A misconfigured worker must refuse to start rather than serve wrong results. Every
test here asserts a specific refusal, because each of these settings can silently
break parity or security if it drifts.
"""
from __future__ import annotations

import dataclasses

import pytest

from app.config import (
    PINNED_MODEL_NAME,
    PROXY_TIMEOUT_CEILING_SECONDS,
    ConfigError,
    WorkerConfig,
)


def valid_config(**overrides) -> WorkerConfig:
    base = dict(
        api_key="k" * 40,
        model_name=PINNED_MODEL_NAME,
        allowed_url_hosts=["s3.us-west-004.backblazeb2.com"],
    )
    base.update(overrides)
    return WorkerConfig(**base)


def test_a_valid_config_passes():
    valid_config().validate()


class TestRequiredSecrets:
    def test_missing_api_key_is_fatal(self):
        """The /process endpoint is publicly reachable through the RunPod proxy.
        Starting without auth is not a degraded mode, it is an open door."""
        with pytest.raises(ConfigError, match="GPU_WORKER_API_KEY"):
            valid_config(api_key="").validate()

    def test_implausibly_short_api_key_is_rejected(self):
        with pytest.raises(ConfigError, match="implausibly short"):
            valid_config(api_key="short").validate()

    def test_missing_model_name_is_fatal(self):
        with pytest.raises(ConfigError, match="MODEL_NAME"):
            valid_config(model_name="").validate()


class TestModelPin:
    """The model is a CONSTRAINT, enforced at four layers.

    buffalo_l (w600k_r50) and buffalo_s (w600k_mbf) are different 512-dim
    embedding spaces. Both produce valid-looking unit vectors, and only buffalo_l
    matches India's FAISS index — so a mismatch fails SILENTLY. Enforced here, in
    burst/config.py, in the Dockerfile build args, and in CI.
    """

    def test_the_pin_is_buffalo_l(self):
        from app.config import PINNED_MODEL_NAME

        assert PINNED_MODEL_NAME == "buffalo_l"

    def test_the_default_is_the_pin(self):
        from app.config import PINNED_MODEL_NAME

        assert WorkerConfig(api_key="k" * 40).model_name == PINNED_MODEL_NAME

    def test_buffalo_s_is_refused(self):
        """The specific mistake this exists to prevent."""
        with pytest.raises(ConfigError, match="PINNED"):
            valid_config(model_name="buffalo_s").validate()

    @pytest.mark.parametrize(
        "model", ["buffalo_s", "buffalo_m", "buffalo_sc", "antelopev2", "BUFFALO_L"]
    )
    def test_every_other_value_is_refused(self, model):
        with pytest.raises(ConfigError):
            valid_config(model_name=model).validate()

    def test_the_error_points_at_the_build_arg(self):
        """The model is baked in at build time, so a runtime disagreement means
        the IMAGE is wrong — the operator needs to be told that, not to go
        hunting through runtime environment variables."""
        with pytest.raises(ConfigError) as exc:
            valid_config(model_name="buffalo_s").validate()
        assert "build-arg" in str(exc.value)

    def test_an_env_override_is_detected_not_silently_forced(self, monkeypatch):
        """Read so validate() can reject it. Silently forcing the pin would be
        worse: the operator would believe their setting took effect."""
        from app.config import load_config

        monkeypatch.setenv("GPU_WORKER_API_KEY", "k" * 40)
        monkeypatch.setenv("MODEL_NAME", "buffalo_s")
        monkeypatch.setenv("ALLOWED_URL_HOSTS", "s3.example.com")

        loaded = load_config()
        assert loaded.model_name == "buffalo_s"
        with pytest.raises(ConfigError, match="PINNED"):
            loaded.validate()


class TestSsrfGuard:
    def test_empty_allowlist_is_fatal(self):
        """Refusing to start is better than starting as an open fetcher. The
        fetcher also fails closed, so this is defence in depth."""
        with pytest.raises(ConfigError, match="ALLOWED_URL_HOSTS"):
            valid_config(allowed_url_hosts=[]).validate()


class TestTimeoutLadder:
    def test_deadline_must_be_below_the_proxy_ceiling(self):
        """Above 100s the response is lost to a Cloudflare 524, which the client
        cannot interpret. Better to return a partial batch at 75s."""
        with pytest.raises(ConfigError, match="proxy"):
            valid_config(
                request_deadline_seconds=PROXY_TIMEOUT_CEILING_SECONDS
            ).validate()

    def test_default_deadline_leaves_headroom(self):
        config = valid_config()
        assert config.request_deadline_seconds < PROXY_TIMEOUT_CEILING_SECONDS
        assert PROXY_TIMEOUT_CEILING_SECONDS - config.request_deadline_seconds >= 20

    def test_download_timeout_must_be_below_the_deadline(self):
        with pytest.raises(ConfigError, match="DOWNLOAD_TIMEOUT_SECONDS"):
            valid_config(
                download_timeout_seconds=80, request_deadline_seconds=75
            ).validate()


class TestParityValues:
    def test_embedding_dim_must_be_512(self):
        """The .bin format is a raw float32[512] dump and faiss_updater rejects
        anything else."""
        with pytest.raises(ConfigError, match="embedding_dim"):
            valid_config(embedding_dim=256).validate()

    def test_det_thresh_must_be_a_probability(self):
        with pytest.raises(ConfigError, match="DET_THRESH"):
            valid_config(det_thresh=1.5).validate()
        with pytest.raises(ConfigError, match="DET_THRESH"):
            valid_config(det_thresh=0.0).validate()

    def test_defaults_match_indias_cpu_path(self):
        """These four values decide which pixels reach the model."""
        config = valid_config()
        assert config.det_size == (640, 640)
        assert config.det_thresh == 0.6
        assert config.max_detection_size == 640
        assert config.embedding_dim == 512

    def test_ctx_id_defaults_to_gpu(self):
        """0 = first GPU. India's CPU path uses -1; this is the ONE intentional
        difference between the two."""
        assert valid_config().ctx_id == 0

    def test_provider_defaults_to_cuda(self):
        assert valid_config().onnx_provider == "CUDAExecutionProvider"


class TestLimits:
    def test_batch_size_must_be_positive(self):
        with pytest.raises(ConfigError, match="MAX_BATCH_IMAGES"):
            valid_config(max_batch_images=0).validate()

    def test_multiple_problems_are_reported_together(self):
        """One restart per problem would be a miserable way to configure a Pod
        that costs money while it boots."""
        with pytest.raises(ConfigError) as exc:
            WorkerConfig(api_key="", model_name="", allowed_url_hosts=[]).validate()
        message = str(exc.value)
        assert "GPU_WORKER_API_KEY" in message
        assert "MODEL_NAME" in message
        assert "ALLOWED_URL_HOSTS" in message


class TestRedaction:
    def test_api_key_is_never_rendered(self):
        rendered = valid_config().redacted()
        assert "k" * 40 not in str(rendered)
        assert rendered["api_key"] == "<set:40 chars>"

    def test_redacted_still_shows_the_parity_values(self):
        """The startup log needs to be enough to diagnose a parity mismatch."""
        rendered = valid_config().redacted()
        for key in ("model_name", "det_size", "det_thresh", "max_detection_size",
                    "onnx_provider", "embedding_dim"):
            assert key in rendered
