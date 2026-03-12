from services.visual_models import VisualState
from services.visual_memory_filter import VisualMemoryFilter
from services.memory_engine import MemoryEngine
from models.memory import VisualMemory, EpisodicMemoryRecord


class MultimodalMemoryEngine:
    """
    Stores distilled visual + contextual memory.
    """

    def __init__(self, memory_engine: MemoryEngine):
        self.memory = memory_engine
        self.filter = VisualMemoryFilter()

    def remember_visual_context(
        self,
        visual_state: VisualState,
        intent_type: str,
        confidence: float
    ):
        salient_texts = self.filter.extract_salient_text(visual_state)
        if not salient_texts:
            return

        visual_mem = VisualMemory(
            active_window=visual_state.active_window,
            salient_text=" | ".join(salient_texts[:5]),
            intent_type=intent_type,
            confidence=confidence
        )

        self.memory.remember_visual(visual_mem)

    def remember_episode(self, summary: str, metadata: dict):
        episode = EpisodicMemoryRecord(
            summary=summary,
            metadata=metadata
        )
        self.memory.remember_episode(episode)
