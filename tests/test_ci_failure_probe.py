"""Temporary controlled failure for issue #19's hosted CI verification."""


def test_controlled_ci_failure():
    assert False, "Intentional issue #19 CI probe; remove after observing red check"
