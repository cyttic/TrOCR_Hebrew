from PIL import Image, ImageOps
import numpy as np
import torch


class HebrewBlockProcessor:
    """
    Adapted from HATFormer BlockProcessor (Chan et al., 2025) for Hebrew RTL text.

    Problem with standard TrOCR processor: it squishes the full text line into
    384x384, destroying the aspect ratio and blurring vertical strokes.

    Solution: standardize height to 64px, then tile the image left-to-right
    top-to-bottom inside the 384x384 ViT container — up to 2304px of text width
    across 6 rows. Remaining space is padded white.

    Steps:
      1. Flip horizontally — Hebrew is RTL, ViT reads LTR. Flipping aligns
         sentence start with early positional embeddings.
      2. Resize height to 64px (4 x ViT patch size of 16px), keep aspect ratio.
      3. Tile into 384x384 container row by row.
      4. Normalize to [-1, 1] (same convention as TrOCR/BEiT).
    """

    TARGET_HEIGHT = 64      # 4 × ViT patch size (16px)
    CONTAINER_SIZE = 384    # ViT input resolution
    IMAGE_MEAN = [0.5, 0.5, 0.5]
    IMAGE_STD  = [0.5, 0.5, 0.5]

    def __call__(self, images, return_tensors="pt"):
        if not isinstance(images, list):
            images = [images]
        pixel_values = torch.stack([self._process(img) for img in images])
        return {"pixel_values": pixel_values}

    def _process(self, image):
        image = image.convert("RGB")

        # step 1 — flip RTL -> LTR
        image = ImageOps.mirror(image)

        # step 2 — resize height to TARGET_HEIGHT, preserve aspect ratio
        w, h = image.size
        new_w = max(1, round(w * self.TARGET_HEIGHT / h))
        image = image.resize((new_w, self.TARGET_HEIGHT), Image.LANCZOS)

        # step 3 — tile into container left-to-right, top-to-bottom
        container = Image.new("RGB", (self.CONTAINER_SIZE, self.CONTAINER_SIZE), (255, 255, 255))
        img_arr = np.array(image)
        src_x, dest_x, dest_y = 0, 0, 0

        while src_x < new_w and dest_y < self.CONTAINER_SIZE:
            chunk_w = min(new_w - src_x, self.CONTAINER_SIZE - dest_x)
            chunk = Image.fromarray(img_arr[:, src_x : src_x + chunk_w])
            container.paste(chunk, (dest_x, dest_y))
            src_x  += chunk_w
            dest_x += chunk_w
            if dest_x >= self.CONTAINER_SIZE:
                dest_x  = 0
                dest_y += self.TARGET_HEIGHT

        # step 4 — normalize to [-1, 1]
        t = torch.tensor(np.array(container), dtype=torch.float32).permute(2, 0, 1) / 255.0
        mean = torch.tensor(self.IMAGE_MEAN).view(3, 1, 1)
        std  = torch.tensor(self.IMAGE_STD).view(3, 1, 1)
        return (t - mean) / std
