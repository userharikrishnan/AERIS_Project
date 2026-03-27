from services.visual_models import VisualState
from typing import List


class VisualMemoryFilter:
    """
    Selects ONLY salient visual information.
    """

    MIN_TEXT_LENGTH = 3
    MAX_TEXT_LENGTH = 80

    def extract_salient_text(self, visual_state: VisualState) -> List[str]:
        salient = []

        for block in visual_state.text_blocks:
            text = block.text.strip()
            if not text:
                continue

            if self.MIN_TEXT_LENGTH <= len(text) <= self.MAX_TEXT_LENGTH:
                salient.append(text)

        return salient
