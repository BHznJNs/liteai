class SkillException(Exception): ...

class InvalidSkillArchiveError(SkillException):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
