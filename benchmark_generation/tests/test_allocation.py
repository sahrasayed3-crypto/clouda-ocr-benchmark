from ocrbench.allocation import allocate


def test_allocation_is_deterministic_and_matches_configured_total():
    config = {
        "default_severity_mix": {"light": 1, "medium": 1, "heavy": 1},
        "buckets": {"blur": {"count": 5, "distortions": ["a", "b"]}},
        "combined": {"total": 2, "profiles": {"p": 2}},
    }
    ids = [f"s{i}" for i in range(4)]
    first = allocate(ids, config, 17)
    assert first == allocate(ids, config, 17)
    assert len(first) == 7
    assert len({item["distorted_id"] for item in first}) == 7

