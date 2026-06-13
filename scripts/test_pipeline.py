"""
Tests for the SteganoPipeline class (string-level transformations).

Run with::

    cd /home/madict/kDrive/UNIFI/TESI/meteor_image_generator
    conda run -n taming python3 -m scripts.test_pipeline
"""

from scripts.pipeline import SteganoPipeline, build_pipeline_from_options


def test_with_xor():
    """RS correction code + XOR mask roundtrip."""
    pipe = SteganoPipeline(xor_key=42, char_encoding="ASCII")
    original = "Hello World!"
    protected = pipe.encode_message(original)
    recovered = pipe.decode_message(protected)
    assert original == recovered, (
        f"RS+XOR roundtrip failed:\n"
        f"  original:  {original!r}\n"
        f"  recovered: {recovered!r}"
    )
    print("[PASS] test_with_xor")


def test_no_xor():
    """RS correction code without XOR mask (xor_key=0 is effectively identity)."""
    pipe = SteganoPipeline(xor_key=0, char_encoding="ASCII")
    original = "Test123"
    protected = pipe.encode_message(original)
    recovered = pipe.decode_message(protected)
    assert original == recovered, (
        f"RS no-XOR roundtrip failed:\n"
        f"  original:  {original!r}\n"
        f"  recovered: {recovered!r}"
    )
    print("[PASS] test_no_xor")


def test_unicode_roundtrip():
    """Unicode + RS roundtrip."""
    pipe = SteganoPipeline(xor_key=77, char_encoding="UNICODE")
    original = "Ciao 🌍!"
    protected = pipe.encode_message(original)
    recovered = pipe.decode_message(protected)
    assert original == recovered, (
        f"Unicode+RS roundtrip failed:\n"
        f"  original:  {original!r}\n"
        f"  recovered: {recovered!r}"
    )
    print("[PASS] test_unicode_roundtrip")


def test_empty_message():
    """Empty message should roundtrip cleanly."""
    pipe = SteganoPipeline(xor_key=42, char_encoding="ASCII")
    assert pipe.encode_message("") == ""
    assert pipe.decode_message("") == ""
    print("[PASS] test_empty_message")


def test_build_from_options():
    """build_pipeline_from_options should return a working pipeline."""
    class FakeOptions:
        xor_key = 123
        char_encoding = "ASCII"

    opts = FakeOptions()
    pipe = build_pipeline_from_options(opts)
    original = "FromOptions"
    protected = pipe.encode_message(original)
    recovered = pipe.decode_message(protected)
    assert original == recovered
    print("[PASS] test_build_from_options")


def test_encoder_decoder_no_ecc_arg():
    """Verify encoder/decoder don't accept error_correction argument anymore."""
    import inspect
    from scripts.encoder import SteganographyEncoder
    from scripts.decoder import SteganographyDecoder

    enc_sig = inspect.signature(SteganographyEncoder.__init__)
    dec_sig = inspect.signature(SteganographyDecoder.__init__)

    for param_name in ("error_correction", "error_correction_method", "ecc"):
        assert param_name not in enc_sig.parameters, (
            f"Encoder still accepts '{param_name}' parameter"
        )
        assert param_name not in dec_sig.parameters, (
            f"Decoder still accepts '{param_name}' parameter"
        )
    print("[PASS] test_encoder_decoder_no_ecc_arg")


if __name__ == "__main__":
    test_with_xor()
    test_no_xor()
    test_unicode_roundtrip()
    test_empty_message()
    test_build_from_options()
    test_encoder_decoder_no_ecc_arg()
    print("\n=== ALL PIPELINE TESTS PASSED ===")