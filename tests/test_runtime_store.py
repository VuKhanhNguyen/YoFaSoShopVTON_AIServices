"""Regression cho các invariant quan trọng của queue manifest, không dùng GPU."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from api.core.errors import ServiceError
from api.core.store import Store


def test_queue_capacity_is_atomic(settings):
    store = Store(settings)
    def submit(index):
        try:
            return store.submit("TRY_ON", f"capacity-{index}", {"test": index}, [], "mock", 1)["id"]
        except ServiceError as exc:
            assert exc.code == "QUEUE_FULL"
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(submit, range(8)))
    assert len([result for result in results if result]) == 1


def test_claims_are_unique_and_fenced(settings):
    store = Store(settings)
    for index in range(6):
        store.submit("TRY_ON", f"claim-{index}", {"index": index}, [], "mock", 20)
    with ThreadPoolExecutor(max_workers=6) as pool:
        claimed = list(pool.map(lambda _: store.claim(), range(6)))
    assert len({job["id"] for job in claimed}) == 6
    job = claimed[0]
    with pytest.raises(ServiceError, match="Job không còn"):
        store.finish(job["id"], claim_token="wrong-claim", result={})
    assert store.job(job["id"])["status"] == "PROCESSING"
    store.finish(job["id"], claim_token=job["claim_token"], result={"ok": True})
    assert store.public_job(store.job(job["id"]))["result"] == {"ok": True}


def test_worker_lock_is_global_to_runtime(settings):
    from workers.runner import worker_lock
    with worker_lock(settings.runtime_dir):
        with pytest.raises(RuntimeError):
            with worker_lock(settings.runtime_dir):
                pass
    with worker_lock(settings.runtime_dir):
        pass
