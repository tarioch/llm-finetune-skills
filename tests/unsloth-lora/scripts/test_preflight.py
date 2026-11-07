import preflight


def test_parse_nvidia_smi_csv():
    out = "0, NVIDIA GeForce RTX 4070 Ti, 1500, 12282\n1, Other, 0, 8000\n"
    assert preflight.parse_csv(out) == [
        ["0", "NVIDIA GeForce RTX 4070 Ti", "1500", "12282"],
        ["1", "Other", "0", "8000"],
    ]


def test_estimate_matches_the_two_measured_runs():
    assert abs(preflight.estimate_gib(3) - 6.6) < 0.1
    assert abs(preflight.estimate_gib(7) - 8.1) < 0.1


def test_no_advice_when_the_estimate_fits():
    assert preflight.batch_advice(free_gib=11.0, params_b=7) is None


def test_advice_lowers_the_batch_size_and_keeps_the_effective_batch():
    advice = preflight.batch_advice(free_gib=7.0, params_b=7)
    assert "--batch-size 2 --grad-accum 8" in advice
    assert "8.1 GiB" in advice
