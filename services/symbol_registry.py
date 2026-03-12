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

    def register(self, symbol: Symbol):
        self._symbols[symbol.name] = symbol

    def resolve(self, intent_type):
        return self._symbols.get(intent_type)
