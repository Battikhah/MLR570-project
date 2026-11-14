from scripts.run_suite import GPU_FAULT, supervise


def test_supervisor_restarts_after_a_gpu_fault_until_a_clean_exit():
    codes, sleeps = [GPU_FAULT, GPU_FAULT, 0], []
    assert (
        supervise(["w"], lambda _m: False, run=lambda *_a, **_k: codes.pop(0), sleep=sleeps.append)
        == 0
    )
    assert codes == [] and len(sleeps) == 2


def test_supervisor_stops_when_the_window_closes():
    calls = []
    out = supervise(
        ["w"],
        lambda _m: True,
        run=lambda *_a, **_k: calls.append(1) or GPU_FAULT,
        sleep=lambda _s: None,
    )
    assert out == GPU_FAULT and calls == [1]
