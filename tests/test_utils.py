import pytest

from strolidlib import utils


def test_seconds_to_ydhms_zero():
    assert utils.seconds_to_ydhms(0) == "0s"


def test_seconds_to_ydhms_combination():
    value = 1 * 86_400 + 2 * 3_600 + 3 * 60 + 4
    assert utils.seconds_to_ydhms(value) == "1d 2h 3m 4s"


def test_are_we_parallel_flag_present():
    assert utils.are_we_parallel({"parallel": True}) is True


def test_are_we_parallel_missing_flag_defaults_false():
    assert utils.are_we_parallel({}) is False


def test_opts_have_changed_detects_difference():
    assert utils.opts_have_changed({"a": 1}, {"a": 2}) is True
    assert utils.opts_have_changed({"a": 1}, {"a": 1}) is False


def test_include_default_opts_preserves_existing_values():
    opts = {"parallel": True}
    defaults = {"parallel": False, "cuda_device": 0}

    result = utils.include_default_opts(opts, defaults)

    assert result is opts
    assert result["parallel"] is True
    assert result["cuda_device"] == 0

