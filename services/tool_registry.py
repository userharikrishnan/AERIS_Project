class ToolRegistry:
    def __init__(self):
        self._tools = {}
        self._capability_map = {}

    def register(self, tool):
        self._tools[tool.name] = tool

        # Register capabilities
        for cap in getattr(tool, "capabilities", []):
            self._capability_map.setdefault(cap, []).append(tool)

    def get(self, name: str):
        return self._tools.get(name)

    def find_by_capability(self, action_name: str):
        tools = self._capability_map.get(action_name, [])
        return tools[0] if tools else None

    def find_alternative(self, action_name: str, exclude=None):
        tools = self._capability_map.get(action_name, [])
        for tool in tools:
            if tool.name != exclude:
                return tool
        return None

    def list_tools(self):
        return list(self._tools.keys())