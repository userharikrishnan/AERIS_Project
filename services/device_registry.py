from services.trust_models import TrustLevel

class Device:
    def __init__(self, device_id, name, trust: TrustLevel, capabilities: list):
        self.device_id = device_id
        self.name = name
        self.trust = trust
        self.capabilities = capabilities
        self.online = True


class DeviceRegistry:
    def __init__(self):
        self.devices = {}

    def register(self, device: Device):
        self.devices[device.device_id] = device

    def get(self, device_id: str):
        return self.devices.get(device_id)

    def all_online(self):
        return [
            d for d in self.devices.values()
            if d.online
        ]

    def capable_devices(self, capability: str):
        return [
            d for d in self.all_online()
            if capability in d.capabilities
        ]
