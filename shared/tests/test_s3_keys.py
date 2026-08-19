from shared.s3.keys import build_key


def test_build_key_layers_type_id_and_name():
    assert build_key("video", "abc", "clip.mp4") == "video/abc/clip.mp4"


def test_build_key_ignores_directory_components_in_the_name():
    # The name is a display label from the client, never a path. Without this a
    # caller could write outside its own prefix.
    assert build_key("video", "abc", "../../etc/passwd") == "video/abc/passwd"
    assert build_key("video", "abc", "/absolute/path.mp4") == "video/abc/path.mp4"


def test_build_key_keeps_the_id_prefix_even_for_a_hostile_name():
    key = build_key("video", "abc", "../../../evil.mp4")
    assert key.startswith("video/abc/")
    assert ".." not in key


def test_build_key_replaces_characters_that_are_awkward_in_a_key():
    assert build_key("video", "abc", "my clip (final).mp4") == "video/abc/my_clip_final_.mp4"


def test_build_key_preserves_the_extension_through_sanitising():
    assert build_key("evidence_frame", "abc", "фото.jpg").endswith(".jpg")


def test_build_key_falls_back_when_nothing_survives_sanitising():
    assert build_key("video", "abc", "???") == "video/abc/file"
    assert build_key("video", "abc", "") == "video/abc/file"


def test_build_key_truncates_a_very_long_name():
    key = build_key("video", "abc", "a" * 500 + ".mp4")
    assert len(key.rsplit("/", 1)[1]) <= 128


def test_the_same_name_under_two_ids_does_not_collide():
    assert build_key("video", "one", "clip.mp4") != build_key("video", "two", "clip.mp4")
