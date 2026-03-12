class Planner:
    def create_plan(self, intent):
        """
        Converts intent into step-by-step actions
        """

        if intent.type == "OPEN_APP":
            return [
                {"step": 1, "action": "open_app", "params": intent.entities}
            ]

        if intent.type == "SEARCH":
            return [
                {"step": 1, "action": "web_search", "params": intent.entities}
            ]

        if intent.type == "CHAT":
            return [
                {"step": 1, "action": "respond", "params": intent.entities}
            ]

        return []
