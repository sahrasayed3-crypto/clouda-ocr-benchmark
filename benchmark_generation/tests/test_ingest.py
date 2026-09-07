from ocrbench.ingest import Selector, choose_nearest


def test_choose_nearest_stays_unused_and_minimizes_distance():
    assert choose_nearest(10, [6, 9, 11, 13], {9}) == 11


def test_selector_parser_preserves_requested_identity():
    s = Selector.parse("owner/data,train,17")
    assert (s.dataset, s.split, s.row) == ("owner/data", "train", 17)

