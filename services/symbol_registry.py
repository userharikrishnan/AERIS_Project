from services.trust_models import ActionSensitivity

class Symbol:
    def __init__(self, name, action, sensitivity: ActionSensitivity):
        self.name = name
        self.action = action
        self.sensitivity = sensitivity


class SymbolRegistry:
    def __init__(self):
        self._symbols = {}

        self.register(
            Symbol(
                name="OPEN_APP",
                action="open_app",
                sensitivity=ActionSensitivity.LOW
            )
        )

        self.register(
            Symbol(
                name="WEB_SEARCH",
                action="web_search",
                sensitivity=ActionSensitivity.LOW
            )
        )

        self.register(
            Symbol(
                name="WEB_NAVIGATE",
                action="web_navigate",
                sensitivity=ActionSensitivity.LOW
            )
        )

        self.register(
            Symbol(
                name="FILE_READ",
                action="file_read",
                sensitivity=ActionSensitivity.MEDIUM
            )
        )

        self.register(
            Symbol(
                name="FILE_WRITE",
                action="file_write",
                sensitivity=ActionSensitivity.MEDIUM
            )
        )

        self.register(
            Symbol(
                name="FILE_DELETE",
                action="file_delete",
                sensitivity=ActionSensitivity.HIGH
            )
        )

        self.register(
            Symbol(
                name="MEMORY_STORE",
                action="memory_store",
                sensitivity=ActionSensitivity.LOW
            )
        )

        self.register(
            Symbol(
                name="MEMORY_RECALL",
                action="memory_recall",
                sensitivity=ActionSensitivity.LOW
            )
        )

        self.register(
            Symbol(
                name="GOAL_CREATE",
                action="goal_create",
                sensitivity=ActionSensitivity.MEDIUM
            )
        )

        self.register(
            Symbol(
                name="ROLLBACK",
                action="rollback",
                sensitivity=ActionSensitivity.HIGH
            )
        )

    def register(self, symbol: Symbol):
        self._symbols[symbol.name] = symbol

    def resolve(self, intent_type):
        return self._symbols.get(intent_type)

    def is_valid(self, action_name: str) -> bool:
        return any(sym.action == action_name for sym in self._symbols.values())

    def get_sensitivity(self, action_name: str):
        for sym in self._symbols.values():
            if sym.action == action_name:
                return sym.sensitivity
        return ActionSensitivity.LOW