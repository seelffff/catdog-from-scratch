from io import BytesIO
import warnings

from fastapi import HTTPException, status
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_PIXELS = 20_000_000
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}


def decode_image(data: bytes) -> Image.Image:
    """Байты загрузки → RGB-изображение, понятные ошибки вместо 500."""
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Пустой файл")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format not in SUPPORTED_FORMATS:
                    raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Поддерживаются JPEG, PNG и WebP")
                if image.width * image.height > MAX_PIXELS:
                    raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "Слишком большое разрешение")
                image.load()
                with ImageOps.exif_transpose(image) as oriented:
                    return oriented.convert("RGB")
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "Слишком большое разрешение") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Не удалось прочитать изображение") from exc
