import make_modelfile


def test_write_modelfile_is_lf_only(tmp_path):
    run = {"chat_format": "chatml", "modelfile_system": "Be brief."}
    out = make_modelfile.write_modelfile(run, tmp_path / "m.gguf", tmp_path / "Modelfile")
    data = out.read_bytes()
    assert b"\r" not in data
    assert data.startswith(f"FROM {tmp_path / 'm.gguf'}\n".encode())
    assert data.endswith(b'SYSTEM """Be brief."""\n')
