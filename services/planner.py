class Planner:
    def create_plan(self, intent, context=None):
        """
        Converts intent into step-by-step action plan.
        Every intent type the NLP can produce has a corresponding plan.
        Now supports:
        - Multi-step chaining
        - Confidence-aware planning
        - Priority scoring
        - Fallback strategies
        """
        t = intent.type
        e = intent.entities

        # -------------------------
        # Confidence check
        # -------------------------
        # Only clarify if BOTH confidence is very low AND margin is weak
        # (high margin = classifier is sure even with low absolute confidence)
        low_conf   = hasattr(intent, 'confidence') and intent.confidence < 0.18
        low_margin = hasattr(intent, 'margin') and intent.margin < 0.08
        if low_conf and low_margin:
            return [{
                "step": 1,
                "action": "clarify",
                "params": {"question": "Can you clarify your request?"},
                "priority": 0.9
            }]

        # -------------------------
        # App control
        # -------------------------
        if t == "OPEN_APP":
            return [{"step": 1, "action": "open_app", "params": e, "priority": 0.8}]

        if t == "CLOSE_APP":
            return [{"step": 1, "action": "close_app", "params": e, "priority": 0.8}]

        # -------------------------
        # Web
        # -------------------------
        if t == "WEB_SEARCH":
            plan = [
                {"step": 1, "action": "web_search", "params": e, "priority": 0.8}
            ]

            # Multi-step: if user implies saving results
            raw_text = getattr(intent, 'raw_text', '')
            if raw_text and ("save" in raw_text.lower() or "store" in raw_text.lower() or "write" in raw_text.lower()):
                plan.append({
                    "step": 2,
                    "action": "file_write",
                    "params": {"content": "search_results", "filename": "search_results.txt"},
                    "priority": 0.7
                })

            return plan

        if t == "WEB_NAVIGATE":
            return [{"step": 1, "action": "web_navigate", "params": e, "priority": 0.8}]

        if t == "WEB_SCRAPE":
            plan = [{"step": 1, "action": "web_scrape", "params": e, "priority": 0.85}]
            # Auto-chain report if user implied saving
            raw_text = getattr(intent, 'raw_text', '')
            if raw_text and any(w in raw_text.lower() for w in ['save', 'report', 'store', 'write', 'file', 'generate']):
                plan.append({
                    "step": 2,
                    "action": "generate_report",
                    "params": {"format": "md", "save_path": "desktop", "title": "Scraped Data"},
                    "priority": 0.75
                })
            return plan

        if t == "GENERATE_REPORT":
            return [{"step": 1, "action": "generate_report", "params": e, "priority": 0.8}]

        if t == "SYSTEM_INFO":
            return [{"step": 1, "action": "system_info", "params": e, "priority": 0.7}]

        if t == "SCREENSHOT":
            return [{"step": 1, "action": "screenshot", "params": e, "priority": 0.75}]



        # -------------------------
        # File system
        # -------------------------
        if t == "FILE_READ":
            return [{"step": 1, "action": "file_read", "params": e, "priority": 0.7}]

        if t == "FILE_WRITE":
            return [{"step": 1, "action": "file_write", "params": e, "priority": 0.7}]

        if t == "FILE_DELETE":
            return [{"step": 1, "action": "file_delete", "params": e, "priority": 0.6}]

        if t == "FILE_LIST":
            return [{"step": 1, "action": "file_list", "params": e, "priority": 0.6}]

        # -------------------------
        # Memory
        # -------------------------
        if t == "MEMORY_STORE":
            return [{"step": 1, "action": "memory_store", "params": e, "priority": 0.7}]

        if t == "MEMORY_RECALL":
            return [{"step": 1, "action": "memory_recall", "params": e, "priority": 0.7}]

        if t == "MEMORY_FORGET":
            return [{"step": 1, "action": "memory_forget", "params": e, "priority": 0.6}]

        # -------------------------
        # Goals
        # -------------------------
        if t == "GOAL_CREATE":
            return [{"step": 1, "action": "goal_create", "params": e, "priority": 0.8}]

        if t == "GOAL_LIST":
            return [{"step": 1, "action": "goal_list", "params": e, "priority": 0.6}]

        if t == "GOAL_PAUSE":
            return [{"step": 1, "action": "goal_pause", "params": e, "priority": 0.7}]

        if t == "GOAL_RESUME":
            return [{"step": 1, "action": "goal_resume", "params": e, "priority": 0.7}]

        if t == "GOAL_COMPLETE":
            return [{"step": 1, "action": "goal_complete", "params": e, "priority": 0.7}]

        # -------------------------
        # Vision / Screen
        # -------------------------
        if t in {"VISION_QUERY", "READ_SCREEN"}:
            return [{"step": 1, "action": "screen_read", "params": e, "priority": 0.8}]

        if t == "ACTIVE_WINDOW":
            return [{"step": 1, "action": "active_window", "params": e, "priority": 0.6}]

        if t == "LIST_WINDOWS":
            return [{"step": 1, "action": "list_windows", "params": e, "priority": 0.6}]

        # -------------------------
        # Rollback / confirm / cancel
        # -------------------------
        if t == "ROLLBACK":
            return [{"step": 1, "action": "rollback", "params": e, "priority": 0.8}]

        if t == "CONFIRM":
            return [{"step": 1, "action": "confirm", "params": e, "priority": 0.9}]

        if t == "CANCEL":
            return [{"step": 1, "action": "cancel", "params": e, "priority": 0.8}]

        # -------------------------
        # Reasoning / identity
        # -------------------------
        if t == "REASONING":
            return [{"step": 1, "action": "reason", "params": e, "priority": 0.7}]

        if t == "IDENTITY_QUERY":
            return [{"step": 1, "action": "identity_query", "params": e, "priority": 0.9}]

        # -------------------------
        # Chat / fallback
        # -------------------------
        if t == "CHAT":
            return [{"step": 1, "action": "respond", "params": e, "priority": 0.7}]

        # -------------------------
        # Unknown intent — enhanced fallback strategy
        # Try to reason before responding
        # -------------------------
        return [
            {"step": 1, "action": "reason", "params": e, "priority": 0.6},
            {"step": 2, "action": "respond", "params": e, "priority": 0.5}
        ]

    def expand_goal(self, description: str):
        """
        Breaks a goal description into sub-objectives.
        Used by /core/goals/create endpoint.
        Now uses intent-driven expansion for smarter branching.
        """
        description_lower = description.lower()
        
        # Intent-driven expansion based on description content
        is_web_related = any(word in description_lower for word in ["browse", "search", "find", "look", "research", "google", "web", "online", "site", "url"])
        is_file_related = any(word in description_lower for word in ["file", "document", "write", "create", "save", "edit", "folder", "directory"])
        is_app_related = any(word in description_lower for word in ["open", "launch", "start", "run", "application", "program", "app"])

        if is_web_related:
            return [
                "Open the browser",
                "Search for the topic",
                "Review the results",
                "Extract relevant information",
                "Summarize findings"
            ]

        if is_file_related:
            return [
                "Create or locate the file",
                "Write the content",
                "Save the document",
                "Verify file integrity"
            ]

        if is_app_related:
            return [
                "Identify the target application",
                "Launch the application",
                "Confirm it is running",
                "Verify accessibility"
            ]

        # Generic fallback objectives
        return [
            f"Understand: {description}",
            f"Plan execution for: {description}",
            f"Execute: {description}",
            "Verify completion"
        ]