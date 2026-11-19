"""Small CPU helpers for rendering the geometric curriculum reliably."""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def close_binary_mask(mask, kernel_size=3):
    """Close tiny one-pixel gaps in a predicted foreground mask on CPU."""
    mask = np.asarray(mask, dtype=bool)
    kernel_size = int(kernel_size)
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer of at least 3")
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    closed = image.filter(ImageFilter.MaxFilter(kernel_size)).filter(
        ImageFilter.MinFilter(kernel_size)
    )
    return np.asarray(closed) > 0


def fill_binary_mask_holes(mask):
    """Fill enclosed background gaps in a small binary shape mask using CPU flood fill."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask.copy()
    height, width = mask.shape
    exterior = np.zeros_like(mask)
    stack = []

    for x in range(width):
        if not mask[0, x]:
            exterior[0, x] = True
            stack.append((0, x))
        if not mask[height - 1, x] and not exterior[height - 1, x]:
            exterior[height - 1, x] = True
            stack.append((height - 1, x))
    for y in range(height):
        if not mask[y, 0] and not exterior[y, 0]:
            exterior[y, 0] = True
            stack.append((y, 0))
        if not mask[y, width - 1] and not exterior[y, width - 1]:
            exterior[y, width - 1] = True
            stack.append((y, width - 1))

    while stack:
        y, x = stack.pop()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if (0 <= ny < height and 0 <= nx < width
                    and not mask[ny, nx] and not exterior[ny, nx]):
                exterior[ny, nx] = True
                stack.append((ny, nx))
    return ~exterior


def resize_binary_mask(mask, target_extent):
    """Resize a foreground mask to a target maximum bbox extent, preserving its centroid."""
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.where(mask)
    if not len(xs):
        return mask.copy()

    height, width = mask.shape
    source_width = int(xs.max() - xs.min() + 1)
    source_height = int(ys.max() - ys.min() + 1)
    extent = max(source_width, source_height)
    target_extent = int(np.clip(round(target_extent), 1, min(height, width)))
    scale = target_extent / extent
    target_width = max(1, min(width, int(round(source_width * scale))))
    target_height = max(1, min(height, int(round(source_height * scale))))

    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    source_y = np.rint(np.linspace(0, crop.shape[0] - 1, target_height)).astype(np.intp)
    source_x = np.rint(np.linspace(0, crop.shape[1] - 1, target_width)).astype(np.intp)
    # Endpoint-aligned nearest sampling retains the mask's extreme rows and
    # columns, which PIL's nearest resize can drop for very small silhouettes.
    resized = crop[np.ix_(source_y, source_x)]
    local_y, local_x = np.where(resized)
    if not len(local_x):
        return mask.copy()

    center_x, center_y = float(xs.mean()), float(ys.mean())
    left = int(round(center_x - float(local_x.mean())))
    top = int(round(center_y - float(local_y.mean())))
    left = int(np.clip(left, 0, width - target_width))
    top = int(np.clip(top, 0, height - target_height))
    rendered = np.zeros_like(mask)
    rendered[top:top + target_height, left:left + target_width] = resized
    return rendered


def render_canonical_shape(shape_id, center, target_extent, image_size):
    """Rasterize an exact curriculum primitive centered on a sampled object center."""
    extent = int(np.clip(round(target_extent), 1, image_size))
    primitive = Image.new("1", (extent, extent), 0)
    draw = ImageDraw.Draw(primitive)
    bounds = (0, 0, extent - 1, extent - 1)
    if shape_id == 0:
        draw.ellipse(bounds, fill=1)
    elif shape_id == 1:
        draw.rectangle(bounds, fill=1)
    elif shape_id == 2:
        draw.polygon(((extent // 2, 0), (0, extent - 1), (extent - 1, extent - 1)), fill=1)
    else:
        raise ValueError(f"Unsupported canonical shape id: {shape_id}")

    local = np.asarray(primitive, dtype=bool)
    ys, xs = np.where(local)
    canvas = np.zeros((image_size, image_size), dtype=bool)
    if not len(xs):
        return canvas
    left = int(round(float(center[0]) - float(xs.mean())))
    top = int(round(float(center[1]) - float(ys.mean())))
    left = int(np.clip(left, 0, image_size - extent))
    top = int(np.clip(top, 0, image_size - extent))
    canvas[top:top + extent, left:left + extent] = local
    return canvas


def render_antialiased_shape(shape_id, center, target_extent, image_size, supersample=4):
    """Render a canonical shape as pixel coverage for smooth small-size edges."""
    supersample = int(supersample)
    if supersample < 2:
        raise ValueError("supersample must be at least 2")
    high_size = int(image_size) * supersample
    high_center = tuple((float(value) + 0.5) * supersample - 0.5 for value in center)
    high_mask = render_canonical_shape(
        shape_id, high_center, float(target_extent) * supersample, high_size
    )
    coverage = Image.fromarray(high_mask.astype(np.uint8) * 255, mode="L")
    coverage = coverage.resize((int(image_size), int(image_size)), Image.Resampling.LANCZOS)
    return np.asarray(coverage, dtype=np.float32).clip(0.0, 255.0) / 255.0
