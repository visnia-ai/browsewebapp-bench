class BenchmarkError(Exception):
    """Base error for controlled benchmark failures."""


class CatalogError(BenchmarkError):
    """The task catalog is malformed or internally inconsistent."""


class InvalidEnvironmentError(BenchmarkError):
    """The task could not be evaluated because its environment is invalid."""


class PolicyBlockedError(BenchmarkError):
    """Execution was stopped because the target is no longer policy-safe."""


class ExecutorError(BenchmarkError):
    """The agent executor failed independently of the target environment."""


class JudgeError(InvalidEnvironmentError):
    """The independent judge failed, so the agent result cannot be scored."""


class CleanupError(InvalidEnvironmentError):
    """The environment could not prove that attempt state was removed."""
