
class RunStatus:
    FAILED: "RunStatus"
    FINISHED: "RunStatus"

    @staticmethod
    def to_string(status: "RunStatus") -> str: ...
