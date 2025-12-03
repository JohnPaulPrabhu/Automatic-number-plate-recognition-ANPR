from typing import Tuple, Optional

import cv2
import numpy as np
import easyocr


class OCREngine:
    """
    OCR wrapper for reading license plates using EasyOCR.
    """

    def __init__(self, languages=("en",), gpu: bool | None = None):
        """
        :param languages: Tuple of language codes, e.g. ('en',)
        :param gpu: True/False/None (None = auto)
        """
        if gpu is None:
            # EasyOCR auto GPU if available
            gpu = True

        print(f"[OCREngine] Initializing EasyOCR with languages={languages}, gpu={gpu}")
        self.reader = easyocr.Reader(list(languages), gpu=gpu)

    def read_plate(self, plate_img_bgr) -> Tuple[Optional[str], float]:
        """
        Run OCR on a cropped plate image (BGR, uint8).
        Returns:
            (text, confidence) or (None, 0.0) if nothing detected.
        """
        # Convert BGR -> grayscale (OCR often works fine in gray)
        gray = cv2.cvtColor(plate_img_bgr, cv2.COLOR_BGR2GRAY)

        # Optional preprocessing: resize to a stable size
        h, w = gray.shape
        scale = 2.0 if max(h, w) < 80 else 1.0
        if scale != 1.0:
            gray = cv2.resize(
                gray,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_LINEAR,
            )

        # EasyOCR expects RGB or grayscale; we can pass gray directly
        results = self.reader.readtext(gray)

        if not results:
            return None, 0.0

        # results: list of (bbox, text, confidence)
        # We'll choose the text with the highest confidence
        best_text = None
        best_conf = 0.0

        for _, text, conf in results:
            if conf > best_conf:
                best_conf = conf
                best_text = text

        # Clean text (optional: strip spaces)
        if best_text is not None:
            best_text = best_text.strip().replace(" ", "")

        return best_text, float(best_conf)
