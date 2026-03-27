import pytesseract
from PIL import Image
from services.visual_models import TextBlock


class OCREngine:
    """
    Text extraction only.
    """

    def extract(self, image: Image.Image) -> list[TextBlock]:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        blocks = []
        n = len(data["text"])

        for i in range(n):
            text = data["text"][i].strip()
            if not text:
                continue

            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]

            blocks.append(TextBlock(text=text, bbox=(x, y, w, h)))

        return blocks
