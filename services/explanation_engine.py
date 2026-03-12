from typing import Optional
from models.rollback import RollbackRecord


class ExplanationEngine:
    """
    Converts failures, rollbacks, and system decisions
    into human-readable explanations.
    """

    def explain_failure(
        self,
        action: str,
        reason: str,
        rollback: Optional[RollbackRecord],
        confidence: float
    ) -> dict:
        explanation = {
            "summary": "",
            "details": [],
            "rollback_performed": False,
            "suggestion": ""
        }

        # -------------------------
        # Failure summary
        # -------------------------
        explanation["summary"] = (
            f"The action '{action}' could not be completed."
        )

        explanation["details"].append(
            f"Reason: {reason}"
        )

        # -------------------------
        # Confidence signal
        # -------------------------
        if confidence < 0.4:
            explanation["details"].append(
                "System confidence was low for this action."
            )

        # -------------------------
        # Rollback explanation
        # -------------------------
        if rollback:
            explanation["rollback_performed"] = True
            explanation["details"].append(
                f"All reversible changes were undone safely."
            )
            explanation["details"].append(
                f"Rollback action: {rollback.action}"
            )
        else:
            explanation["details"].append(
                "No rollback was required."
            )

        # -------------------------
        # Suggested next step
        # -------------------------
        if confidence < 0.4:
            explanation["suggestion"] = (
                "You may want to clarify the request or provide more details."
            )
        else:
            explanation["suggestion"] = (
                "You can retry the action or choose an alternative approach."
            )

        return explanation
