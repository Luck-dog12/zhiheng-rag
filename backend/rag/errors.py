class AppError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        self.code = code
        self.message = message
        self.status = status
        super().__init__(message)


class LeaseLost(AppError):
    def __init__(self):
        super().__init__("lease_lost", "任务租约已失效", 409)


class GenerationInterrupted(AppError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__("generation_incomplete", "回答尚未完成，可以继续恢复生成", 502)
