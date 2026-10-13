"""
MNIST Preprocessing Pipeline
Prepares raw canvas drawings into MNIST-standard 28x28 centered tensors using
aspect-ratio scaling, bounding box extraction, and center-of-mass centering.
"""

import base64
import io
import math
from typing import Tuple
import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass, shift


def decode_base64_image(base64_str: str) -> Image.Image:
    """Decodes a base64 or data-URL image string into a PIL Image."""
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    image_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_bytes))


def encode_image_to_base64(img_array: np.ndarray, upscale: int = 4) -> str:
    """
    Converts a 28x28 grayscale numpy array [0.0, 1.0] to a base64 PNG data-url
    scaled up by nearest-neighbor for high-clarity browser preview.
    """
    scaled = (np.clip(img_array, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(scaled, mode="L")
    if upscale > 1:
        pil_img = pil_img.resize(
            (28 * upscale, 28 * upscale),
            resample=Image.Resampling.NEAREST
        )
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def preprocess_canvas_image(
    canvas_input: Image.Image | str,
    threshold: float = 15.0
) -> Tuple[np.ndarray, str, bool]:
    """
    Converts canvas drawing into an MNIST standard 28x28 normalized tensor.

    Steps:
    1. Grayscale & foreground detection (white digit on black background).
    2. Extract bounding box of drawn strokes.
    3. Scale bounding box to fit inside 20x20 while preserving aspect ratio.
    4. Center inside 28x28 frame using center of mass (center of gravity).
    5. Normalize pixel values to range [0.0, 1.0].

    Returns:
        tensor: shape (1, 28, 28, 1), float32 in [0, 1]
        preview_data_url: base64 PNG of the 28x28 processed digit
        is_empty: True if no strokes were detected
    """
    if isinstance(canvas_input, str):
        img = decode_base64_image(canvas_input)
    else:
        img = canvas_input

    # Convert to RGBA first to check transparency
    img = img.convert("RGBA")
    r, g, b, a = img.split()
    rgb = Image.merge("RGB", (r, g, b)).convert("L")
    
    # Composite: If drawing is on transparent canvas with white or dark strokes
    np_alpha = np.array(a, dtype=np.float32)
    np_gray = np.array(rgb, dtype=np.float32)

    # Calculate stroke intensity
    # If transparent canvas, mask by alpha
    if np.any(np_alpha < 255):
        # Foreground is where alpha > 20
        # If pixels are dark on transparent background, invert
        if np.mean(np_gray[np_alpha > 20]) < 128:
            foreground = (255.0 - np_gray) * (np_alpha / 255.0)
        else:
            foreground = np_gray * (np_alpha / 255.0)
    else:
        # Check background brightness: if corners are bright, it's black-on-white
        corner_brightness = (
            np_gray[0, 0] + np_gray[0, -1] + np_gray[-1, 0] + np_gray[-1, -1]
        ) / 4.0
        if corner_brightness > 128:
            foreground = 255.0 - np_gray
        else:
            foreground = np_gray

    foreground = np.clip(foreground, 0.0, 255.0)

    # Check if empty (all background)
    active_mask = foreground > threshold
    if not np.any(active_mask):
        empty_tensor = np.zeros((1, 28, 28, 1), dtype=np.float32)
        empty_preview = encode_image_to_base64(np.zeros((28, 28), dtype=np.float32))
        return empty_tensor, empty_preview, True

    # Find bounding box
    rows = np.any(active_mask, axis=1)
    cols = np.any(active_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Add a small padding to bounding box if possible
    box = foreground[rmin:rmax + 1, cmin:cmax + 1]
    box_h, box_w = box.shape

    # Scale box preserving aspect ratio into maximum 20x20
    max_dim = max(box_h, box_w)
    scale = 20.0 / float(max_dim)
    new_w = max(1, int(round(box_w * scale)))
    new_h = max(1, int(round(box_h * scale)))

    pil_box = Image.fromarray(box.astype(np.uint8), mode="L")
    pil_box_resized = pil_box.resize(
        (new_w, new_h),
        resample=Image.Resampling.BILINEAR
    )
    resized_box = np.array(pil_box_resized, dtype=np.float32)

    # Place resized box in 28x28 canvas initially centered roughly
    canvas_28 = np.zeros((28, 28), dtype=np.float32)
    start_y = (28 - new_h) // 2
    start_x = (28 - new_w) // 2
    canvas_28[start_y:start_y + new_h, start_x:start_x + new_w] = resized_box

    # Center of Mass alignment: MNIST standard centers center-of-mass at (13.5, 13.5)
    cy, cx = center_of_mass(canvas_28)
    if not (math.isnan(cy) or math.isnan(cx)):
        shift_y = 13.5 - cy
        shift_x = 13.5 - cx
        # Clip shift to avoid pushing digit out of canvas
        shift_y = np.clip(shift_y, -5.0, 5.0)
        shift_x = np.clip(shift_x, -5.0, 5.0)
        canvas_28 = shift(canvas_28, [shift_y, shift_x], mode='constant', cval=0.0)

    # Normalize to [0.0, 1.0]
    canvas_28 = np.clip(canvas_28 / 255.0, 0.0, 1.0)
    preview_url = encode_image_to_base64(canvas_28, upscale=4)

    # Prepare model tensor (1, 28, 28, 1)
    tensor = canvas_28.reshape(1, 28, 28, 1).astype(np.float32)
    return tensor, preview_url, False
