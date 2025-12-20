"""Depth estimator adapter supporting Depth Anything v2 and MiDaS fallback.

For Depth Anything v2 (trained in meters), provide model_type='custom' and path to checkpoint.
MiDaS is used as a fallback (outputs relative depth, not meters).
"""
from typing import Optional
import numpy as np
import torch


class DepthEstimator:
    def __init__(self, model_type: str = "midas", checkpoint: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.model_type = model_type
        self.checkpoint = checkpoint
        self.model = None
        self.transform = None
        
        if model_type == "midas":
            try:
                self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device)
                self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
            except Exception:
                self.model = None
        elif model_type == "custom":
            if checkpoint is None:
                raise ValueError("checkpoint path required for custom model")
            # Load custom Depth Anything v2 checkpoint
            self.model = torch.load(checkpoint, map_location=device)
        else:
            raise ValueError("Unsupported model_type: %s" % model_type)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict depth map. For Depth Anything v2 output is in meters."""
        if self.model is None:
            raise RuntimeError("No depth model available. Install dependencies or provide a checkpoint.")
        if self.model_type == "midas":
            img = self.transform(image).to(self.device)
            with torch.no_grad():
                prediction = self.model(img.unsqueeze(0))
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1), size=image.shape[:2], mode="bicubic", align_corners=False
                ).squeeze().cpu().numpy()
            # normalize to 0-1
            prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
            return prediction
        else:
            raise NotImplementedError("Custom depth model inference not implemented. Provide an adapter.")
