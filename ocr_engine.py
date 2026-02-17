"""
OCR Module - Dual engine: PaddleOCR + Tesseract
"""
import os
import cv2
import numpy as np
from PIL import Image
import logging

# Offline mode
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCREngine:
    def __init__(self, use_paddle=True, use_tesseract_fallback=True):
        self.use_paddle = use_paddle
        self.use_tesseract_fallback = use_tesseract_fallback

        if self.use_paddle:
            try:
                import paddle
                paddle.set_device('cpu')
                # NOTE: DO NOT call paddle.enable_static() - PaddleOCR requires dynamic mode

                from paddleocr import PaddleOCR
                self.paddle_ocr = PaddleOCR(
                    lang='fr',
                    use_angle_cls=True,
                    use_gpu=False,
                    show_log=False
                )
                logger.info("PaddleOCR initialized successfully")
            except Exception as e:
                logger.warning(f"PaddleOCR initialization failed: {e}")
                self.use_paddle = False

        if self.use_tesseract_fallback:
            try:
                import pytesseract
                # Windows: set path to Tesseract executable
                pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Alina\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
                # Verify it works
                pytesseract.get_tesseract_version()
                self.pytesseract = pytesseract
                logger.info("Tesseract initialized successfully")
            except Exception as e:
                logger.warning(f"Tesseract initialization failed: {e}")
                self.use_tesseract_fallback = False

    def preprocess_image(self, image):
        """Preprocess image and return RGB numpy array (required by both engines)"""
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Denoise + CLAHE on grayscale, return as RGB
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    def extract_text_paddle(self, image):
        """Extract text using PaddleOCR"""
        try:
            result = self.paddle_ocr.ocr(image, cls=True)
            if not result or result[0] is None:
                return ""
            return "\n".join([line[1][0] for line in result[0]])
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            return None

    def extract_text_tesseract(self, image):
        """Extract text using Tesseract"""
        try:
            pil_image = Image.fromarray(image)
            text = self.pytesseract.image_to_string(pil_image, lang='fra', config='--oem 3 --psm 6')
            return text.strip()
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return None

    def extract_text(self, image, preprocess=True):
        """Main extraction - PaddleOCR with Tesseract fallback"""
        img = self.preprocess_image(image) if preprocess else image
        if self.use_paddle:
            text = self.extract_text_paddle(img)
            if text:
                return text
        if self.use_tesseract_fallback:
            text = self.extract_text_tesseract(img)
            if text:
                return text
        return ""
