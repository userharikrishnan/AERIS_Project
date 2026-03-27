from services.trust_models import TrustLevel

class DeviceModel:
    def __init__(
        self,
        device_id: str,
        name: str,
        trust: TrustLevel,
        capabilities: list
    ):
        self.device_id = device_id
        self.name = name
        self.trust = trust
        self.capabilities = capabilities
