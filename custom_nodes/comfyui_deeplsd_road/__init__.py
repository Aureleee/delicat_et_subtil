"""
ComfyUI DeepLSD Road  (minimal -- road side reconstruction only)
================================================================
Three nodes, used in this order, to turn a road mask into clean straight
border lines for each side of the carriageway
(right-click > Add Node > "DeepLSD Road"):

    Road Contour Extraction      : mask -> every visible contour (defects kept)
    Corrupted Border Detection   : flag contour portions broken by holes
    Local Border Reconstruction  : valid contours -> clean straight borders
                                   (output: border_lines = LINE_SEGMENTS)

Pipeline:
    road_mask -> Road Contour Extraction -> Corrupted Border Detection
              -> Local Border Reconstruction -> border_lines

Install: put this folder in  ComfyUI/custom_nodes/  and restart ComfyUI.
"""

from .road_contour_extraction import RoadContourExtraction
from .corrupted_border_detection import CorruptedBorderDetection
from .local_border_reconstruction import LocalBorderReconstruction

NODE_CLASS_MAPPINGS = {
    "RoadContourExtraction": RoadContourExtraction,
    "CorruptedBorderDetection": CorruptedBorderDetection,
    "LocalBorderReconstruction": LocalBorderReconstruction,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RoadContourExtraction": "Road Contour Extraction",
    "CorruptedBorderDetection": "Corrupted Border Detection",
    "LocalBorderReconstruction": "Local Border Reconstruction",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
