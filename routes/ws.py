"""
AERIS WebSocket Route
Real-time voice interface communication layer.
Wraps the core NLP pipeline for WebSocket transport.
"""
import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
ws_router = APIRouter()


@ws_router.websocket("/ws")
async def aeris_websocket(websocket: WebSocket):
    await websocket.accept()
    logger.info("[WebSocket] Client connected")

    try:
        system = websocket.app.state.system
        nlp         = system.nlp_processor
        reasoning   = system.reasoning_engine
        memory      = system.memory_engine
        command_eng = system.command_engine

        # Shared engines from core module (already initialised)
        from routes.core import (
            _session_engine,
            _smart_confirmation,
            _plan_executor,
        )

        BYPASS_ACTIONS = {
            "respond", "identity_query", "reason",
            "memory_store", "memory_recall", "memory_forget",
            "clarify", "goal_list", "active_window", "list_windows",
        }

        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            text     = msg.get("text", "").strip()
            msg_type = msg.get("type", "input")

            if not text:
                continue

            logger.info(f"[WS] Received: '{text}'")

            # Acknowledge — show thinking state
            await websocket.send_json({"type": "state", "state": "thinking"})

            try:
                session = _session_engine.get_or_create_session()

                # Session-close detection
                if _session_engine.detect_close_intent(text, "") and _session_engine.has_active_session:
                    _session_engine.close_current_session()
                    await websocket.send_json({
                        "type": "response",
                        "response": "Session closed. Talk to you later.",
                        "intent": "SESSION_CLOSE",
                        "session_closed": True,
                    })
                    continue

                # ── NLP ──────────────────────────────────────────────────
                nlp_output  = nlp.process(text)
                intent_type = nlp_output.type

                # ── CHAT / fallback path ─────────────────────────────────
                if nlp_output.mode == "CHAT":
                    response = reasoning.language_engine.generate_from_text(
                        original_text=text,
                        intent_type=intent_type,
                        confidence=nlp_output.confidence,
                        entities=nlp_output.entities,
                    )
                    await websocket.send_json({
                        "type": "response",
                        "response": response or "How can I help?",
                        "intent": intent_type,
                        "confidence": round(nlp_output.confidence, 3),
                    })
                    continue

                # ── Reasoning ────────────────────────────────────────────
                reasoning_result = reasoning.reason(
                    intent=nlp_output,
                    original_text=text,
                    memory=memory,
                )

                action_payload = command_eng.plan(
                    reasoning_output=reasoning_result,
                    context={"confidence": reasoning_result.confidence},
                )

                action_name = action_payload.get("action", "respond") if action_payload else "respond"

                # ── Non-destructive bypass ───────────────────────────────
                if not action_payload or action_name in BYPASS_ACTIONS:
                    response = reasoning.language_engine.generate_from_text(
                        original_text=text,
                        intent_type=intent_type,
                        confidence=nlp_output.confidence,
                        entities=nlp_output.entities,
                    )
                    await websocket.send_json({
                        "type": "response",
                        "response": response or "Understood.",
                        "intent": intent_type,
                        "confidence": round(nlp_output.confidence, 3),
                    })
                    continue

                # ── Smart Confirmation ───────────────────────────────────
                action_params  = action_payload.get("params", {})
                smart_decision = _smart_confirmation.evaluate(action_name, action_params)

                if smart_decision.auto_approved:
                    exec_result = _plan_executor.execute_plan(
                        reasoning_result.plan,
                        initial_context={"original_text": text},
                    )
                    _smart_confirmation.record_approval(
                        action_name, action_params, smart_decision.fingerprint
                    )
                    response = reasoning.language_engine.generate_from_text(
                        original_text=text,
                        intent_type=intent_type,
                        confidence=nlp_output.confidence,
                        entities=nlp_output.entities,
                    )
                    summary = exec_result.to_summary() if hasattr(exec_result, "to_summary") else {}
                    await websocket.send_json({
                        "type": "response",
                        "response": response or "Done.",
                        "intent": intent_type,
                        "action": action_name,
                        "auto_approved": True,
                        "result": summary,
                    })

                else:
                    # Ask for confirmation over WebSocket
                    await websocket.send_json({
                        "type": "confirmation_required",
                        "response": (
                            f"I'll need your confirmation to "
                            f"{action_name.replace('_', ' ')}. Should I proceed?"
                        ),
                        "intent": intent_type,
                        "action": action_name,
                        "params": action_params,
                    })

                    confirm_raw = await websocket.receive_text()
                    confirm_msg = json.loads(confirm_raw)

                    if confirm_msg.get("confirmed", False):
                        exec_result = _plan_executor.execute_plan(
                            reasoning_result.plan,
                            initial_context={"original_text": text},
                        )
                        _smart_confirmation.record_approval(
                            action_name, action_params, smart_decision.fingerprint
                        )
                        response = reasoning.language_engine.generate_from_text(
                            original_text=text,
                            intent_type=intent_type,
                            confidence=nlp_output.confidence,
                            entities=nlp_output.entities,
                        )
                        await websocket.send_json({
                            "type": "response",
                            "response": response or "Done.",
                            "intent": intent_type,
                            "action": action_name,
                        })
                    else:
                        await websocket.send_json({
                            "type": "response",
                            "response": "Understood. Action cancelled.",
                            "intent": "CANCEL",
                        })

            except Exception as exc:
                logger.error(f"[WS] Processing error: {exc}", exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "response": "I encountered an error processing that. Please try again.",
                    "error": str(exc),
                })

    except WebSocketDisconnect:
        logger.info("[WebSocket] Client disconnected")
    except Exception as exc:
        logger.error(f"[WebSocket] Fatal connection error: {exc}", exc_info=True)
