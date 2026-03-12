from pydantic import BaseModel

class Instruction(BaseModel):
    action: str
    target: str
    platform: str = "windows"  # default; can be windows/mac/linux/ios/android
    params: dict = {}
