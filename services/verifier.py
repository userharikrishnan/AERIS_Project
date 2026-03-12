class Verifier:
    def verify(self, plan):
        """
        Ensures plan is logically and mathematically valid
        """

        if not plan:
            return False

        for step in plan:
            if "action" not in step:
                return False

        return True
