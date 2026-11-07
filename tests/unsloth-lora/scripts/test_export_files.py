import hashlib

import export_files


def make_export(tmp_path, gguf_folder="gguf_gguf"):
    """What Unsloth leaves: merged weights in gguf/, the GGUF and a stock Modelfile in gguf_gguf/."""
    run = tmp_path / "run"
    (run / "gguf").mkdir(parents=True)
    (run / "gguf" / "model.safetensors").write_bytes(b"x" * 1000)
    (run / gguf_folder).mkdir(exist_ok=True)
    gguf = run / gguf_folder / "m.Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF data")
    (run / gguf_folder / "Modelfile").write_text("FROM m.Q4_K_M.gguf\n")
    return run, gguf


def test_find_ggufs_next_to_the_output_folder(tmp_path):
    run, gguf = make_export(tmp_path)
    assert export_files.find_ggufs(run / "gguf", []) == [gguf]
    assert export_files.find_ggufs(run / "gguf", ["reported.gguf"]) == [export_files.Path("reported.gguf")]


def test_finish_records_and_tidies_up(tmp_path):
    run, gguf = make_export(tmp_path)
    record, notes = export_files.finish(run, run / "gguf", gguf, "q4_k_m")
    assert record == {
        "path": "gguf/m.Q4_K_M.gguf",
        "quant": "q4_k_m",
        "bytes": 9,
        "sha256": hashlib.sha256(b"GGUF data").hexdigest(),
    }
    assert not (run / "gguf" / "model.safetensors").exists()  # merged weights gone
    assert not (run / "gguf_gguf").exists()  # the GGUF moved to gguf/
    assert (run / "gguf" / gguf.name).read_bytes() == b"GGUF data"
    assert not (run / "gguf" / "Modelfile").exists()  # Unsloth's stock-template Modelfile gone
    assert len(notes) == 3


def test_finish_keeps_merged_on_request(tmp_path):
    run, gguf = make_export(tmp_path)
    record, _ = export_files.finish(run, run / "gguf", gguf, "q4_k_m", keep_merged=True)
    assert (run / "gguf" / "model.safetensors").exists()
    assert record["path"] == "gguf_gguf/m.Q4_K_M.gguf" and gguf.exists()  # nothing to rename onto


def test_finish_never_removes_the_folder_holding_the_gguf(tmp_path):
    run, gguf = make_export(tmp_path, gguf_folder="gguf")  # an Unsloth version writing both into one folder
    record, _ = export_files.finish(run, run / "gguf", gguf, "q4_k_m")
    assert gguf.exists() and record["path"] == "gguf/m.Q4_K_M.gguf"


def test_gguf_outside_the_run_folder_is_recorded_absolute(tmp_path):
    run, _ = make_export(tmp_path)
    elsewhere = tmp_path / "elsewhere.gguf"
    elsewhere.write_bytes(b"GGUF")
    record, _ = export_files.finish(run, run / "gguf", elsewhere, "q4_k_m", keep_merged=True)
    assert record["path"] == str(elsewhere.resolve())
