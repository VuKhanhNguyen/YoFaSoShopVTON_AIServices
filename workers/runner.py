"""Queue worker có singleton lock, khôi phục job bị ngắt và watchdog GPU.

Real pipeline nằm trong child process sống lâu để cache model. Timeout làm parent
terminate child trước khi lấy job tiếp; không để hai inference cùng giữ GPU.
"""
import json
import logging
import multiprocessing as mp
import threading
import time
from contextlib import contextmanager

from api.core.config import Settings
from api.core.errors import ServiceError
from api.core.store import Store

LOGGER = logging.getLogger("yofaso.worker")


def error_data(exc):
    """Chỉ công bố lỗi an toàn; traceback giữ ở log worker, không trả path/token."""
    if isinstance(exc, ServiceError):
        return {"errorCode": exc.code, "message": exc.message, "retryable": exc.retryable, "details": exc.details}
    return {"errorCode": "INFERENCE_FAILED", "message": "Xử lý AI thất bại; kiểm tra log theo jobId.", "retryable": False, "details": {}}


def child_loop(config, connection):
    """Child GPU giữ Pipeline/model giữa các job; không khởi tạo CUDA trước spawn."""
    from api.services.pipeline import Pipeline
    pipeline = Pipeline(Settings(**config))
    while True:
        job = connection.recv()
        if job is None:
            return
        try:
            result = pipeline.run(job)
            # Kiểm tra serialization ngay trong child để parent không chờ vô hạn.
            json.dumps(result, allow_nan=False)
            connection.send((result, None))
        except BaseException as exc:
            LOGGER.exception("job_failed jobId=%s", job["id"])
            connection.send((None, error_data(exc)))


@contextmanager
def worker_lock(root):
    """Khóa GPU cục bộ; API/worker dùng cùng volume trên một host."""
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / "worker.lock").open("a+b")
    try:
        # Không đọc byte đã bị Windows locking khóa bởi worker khác.
        if handle.seek(0, 2) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if __import__("os").name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError("Runtime này đã có worker; không chạy worker thứ hai trên cùng GPU.") from exc
    try:
        yield
    finally:
        handle.close()


class Worker:
    """Worker một job/lượt; không tự retry ảnh sai hoặc quality gate không đạt."""
    def __init__(self, settings, child_target=child_loop):
        """Khởi tạo parent watchdog; child_target có thể thay khi test timeout."""
        self.settings = settings
        self.child_target = child_target
        self.store = Store(settings)
        self.stop = threading.Event()
        self.process = self.connection = None
        self.pipeline = None
        self.lease_connection = None

    def heartbeat(self):
        """Cho health/ready biết worker sống, không đồng nghĩa model GPU đã nghiệm thu."""
        self.write_heartbeat(time.time(), self.settings.ai_mode)

    def write_heartbeat(self, timestamp, mode):
        """Heartbeat ghi nguyên tử vào manifest trong khi worker giữ khóa OS."""
        self.store.heartbeat(timestamp, mode, write=True)

    def shutdown_child(self):
        """Đóng child chắc chắn trước khi cấp GPU cho job tiếp theo."""
        if self.process is not None:
            self.process.terminate()
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def execute(self, job):
        """Mock chạy nhẹ trực tiếp; real ở child có giới hạn thời gian ngoài CUDA."""
        if self.settings.ai_mode == "mock":
            from api.services.pipeline import Pipeline
            if self.pipeline is None:
                self.pipeline = Pipeline(self.settings)
            return self.pipeline.run(job)
        if self.process is None or not self.process.is_alive():
            self.shutdown_child()
            context = mp.get_context("spawn")
            self.connection, child = context.Pipe()
            self.process = context.Process(target=self.child_target, args=(self.settings.model_dump(), child), daemon=True)
            self.process.start()
            child.close()
        self.connection.send(job)
        deadline = time.monotonic() + self.settings.job_timeout_seconds
        while time.monotonic() < deadline and not self.stop.is_set():
            self.heartbeat()
            if self.connection.poll(min(1, max(.01, deadline - time.monotonic()))):
                try:
                    result, error = self.connection.recv()
                except EOFError as exc:
                    self.shutdown_child()
                    raise ServiceError("WORKER_PROCESS_DIED", "Tiến trình GPU đã dừng.", 503, retryable=True) from exc
                if error:
                    raise ServiceError(error["errorCode"], error["message"], details=error.get("details"), retryable=error["retryable"])
                return result
            if not self.process.is_alive():
                self.shutdown_child()
                raise ServiceError("WORKER_PROCESS_DIED", "Tiến trình GPU đã dừng.", 503, retryable=True)
        self.shutdown_child()
        raise ServiceError("JOB_TIMEOUT", "Job quá thời gian; worker đã dừng tiến trình inference.", 503, retryable=True)

    def run_one(self):
        """Xử lý một job đã claim và luôn lưu trạng thái cuối có kiểm soát."""
        job = self.store.claim()
        if not job:
            return False
        started = time.perf_counter()
        try:
            result = self.execute(job)
            result["processingSeconds"] = round(time.perf_counter() - started, 2)
            json.dumps(result, allow_nan=False)
            self.heartbeat()
            self.store.finish(job["id"], claim_token=job["claim_token"], result=result)
        except BaseException as exc:
            LOGGER.exception("job_failed jobId=%s", job["id"])
            self.store.finish(job["id"], claim_token=job["claim_token"], error=error_data(exc))
        return True

    def serve(self):
        """Khôi phục PROCESSING bị ngắt sau crash; không tự chạy lại GPU job cũ."""
        with worker_lock(self.settings.runtime_dir):
            self.store.recover()
            try:
                while not self.stop.is_set():
                    self.heartbeat()
                    if not self.run_one():
                        self.stop.wait(self.settings.worker_poll_seconds)
            finally:
                self.shutdown_child()
                try:
                    self.write_heartbeat(None, None)
                finally:
                    self.lease_connection = None


def main():
    """Entry point: python -m workers.runner, chạy từ thư mục AI Service."""
    logging.basicConfig(level=logging.INFO)
    worker = Worker(Settings())
    try:
        worker.serve()
    except KeyboardInterrupt:
        worker.stop.set()
        worker.shutdown_child()


if __name__ == "__main__":
    main()
