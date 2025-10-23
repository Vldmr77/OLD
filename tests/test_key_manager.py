from scalp_system.security import KeyManager


def test_key_manager_roundtrip():
    manager = KeyManager.generate()
    secret = "super-secret-token"
    encrypted = manager.encrypt(secret)
    assert encrypted != secret
    assert manager.decrypt(encrypted) == secret


def test_key_manager_optional_prefix():
    manager = KeyManager.generate()
    secret = manager.encrypt("abc123")
    wrapped = f"enc:{secret}"
    assert manager.maybe_decrypt(wrapped) == "abc123"
    assert manager.maybe_decrypt("plain") == "plain"
    assert manager.maybe_decrypt(None) is None
