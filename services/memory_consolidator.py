import time
from services.memory_engine import MemoryEngine
from models.memory import LongTermMemoryRecord


class MemoryConsolidator:
    """
    Promotes episodic memories into long-term memory.
    """

    PROMOTION_CONFIDENCE = 0.6

    def __init__(self, memory_engine: MemoryEngine):
        self.memory = memory_engine

    def consolidate(self):
        visual_memories = self.memory.fetch_multimodal("visual")

        for vm in visual_memories:
            if vm["confidence"] < self.PROMOTION_CONFIDENCE:
                continue

            record = LongTermMemoryRecord(
                concept=vm["intent_type"],
                evidence=vm["salient_text"],
                importance=vm["confidence"],
                last_accessed=time.time()
            )

            self.memory.remember_long_term(record)
