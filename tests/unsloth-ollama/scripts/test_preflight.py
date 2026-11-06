import preflight


def test_parse_nvidia_smi_csv():
    out = "0, NVIDIA GeForce RTX 4070 Ti, 1500, 12282\n1, Other, 0, 8000\n"
    assert preflight.parse_csv(out) == [
        ["0", "NVIDIA GeForce RTX 4070 Ti", "1500", "12282"],
        ["1", "Other", "0", "8000"],
    ]
