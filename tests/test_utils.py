"""Tests for strolidlib.utils module."""

import pytest
from strolidlib.utils import (
    seconds_to_ydhms,
    are_we_parallel,
    opts_have_changed,
    include_default_opts,
    is_valid_url,
    is_int,
)


class TestSecondsToYdhms:
    """Tests for seconds_to_ydhms function."""

    def test_zero_seconds(self):
        assert seconds_to_ydhms(0) == "0s"

    def test_seconds_only(self):
        assert seconds_to_ydhms(45) == "45s"

    def test_minutes_and_seconds(self):
        assert seconds_to_ydhms(125) == "2m 5s"

    def test_hours_minutes_seconds(self):
        assert seconds_to_ydhms(3661) == "1h 1m 1s"

    def test_days(self):
        assert seconds_to_ydhms(86400) == "1d"

    def test_years(self):
        assert seconds_to_ydhms(31_536_000) == "1y"

    def test_complex_duration(self):
        # 1 year, 2 days, 3 hours, 4 minutes, 5 seconds
        total = 31_536_000 + (2 * 86_400) + (3 * 3_600) + (4 * 60) + 5
        assert seconds_to_ydhms(total) == "1y 2d 3h 4m 5s"


class TestAreWeParallel:
    """Tests for are_we_parallel function."""

    def test_parallel_true(self):
        assert are_we_parallel({"parallel": True}) is True

    def test_parallel_false(self):
        assert are_we_parallel({"parallel": False}) is False

    def test_parallel_missing(self):
        assert are_we_parallel({}) is False


class TestOptsHaveChanged:
    """Tests for opts_have_changed function."""

    def test_same_opts(self):
        opts = {"key": "value"}
        assert opts_have_changed(opts, opts) is False

    def test_different_opts(self):
        assert opts_have_changed({"key": "value1"}, {"key": "value2"}) is True

    def test_empty_opts(self):
        assert opts_have_changed({}, {}) is False


class TestIncludeDefaultOpts:
    """Tests for include_default_opts function."""

    def test_adds_missing_defaults(self):
        opts = {"a": 1}
        defaults = {"b": 2, "c": 3}
        result = include_default_opts(opts, defaults)
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_does_not_override_existing(self):
        opts = {"a": 1, "b": "custom"}
        defaults = {"b": 2, "c": 3}
        result = include_default_opts(opts, defaults)
        assert result["b"] == "custom"


class TestIsValidUrl:
    """Tests for is_valid_url function."""

    def test_valid_http_url(self):
        assert is_valid_url("http://example.com") is True

    def test_valid_https_url(self):
        assert is_valid_url("https://example.com/path/to/resource") is True

    def test_invalid_url(self):
        assert is_valid_url("not-a-url") is False

    def test_empty_string(self):
        assert is_valid_url("") is False


class TestIsInt:
    """Tests for is_int function."""

    def test_int_returns_true(self):
        assert is_int(42) is True

    def test_zero_returns_true(self):
        assert is_int(0) is True

    def test_negative_int_returns_true(self):
        assert is_int(-10) is True

    def test_string_returns_false(self):
        assert is_int("42") is False

    def test_float_returns_false(self):
        assert is_int(3.14) is False

