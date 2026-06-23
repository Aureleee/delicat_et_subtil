from .mask_to_maps       import NODE_CLASS_MAPPINGS as M1, NODE_DISPLAY_NAME_MAPPINGS as D1
from .centerline_nms     import NODE_CLASS_MAPPINGS as M2, NODE_DISPLAY_NAME_MAPPINGS as D2
from .road_tracker       import NODE_CLASS_MAPPINGS as M3, NODE_DISPLAY_NAME_MAPPINGS as D3
from .edge_generator     import NODE_CLASS_MAPPINGS as M4, NODE_DISPLAY_NAME_MAPPINGS as D4
from .tensor_voting      import NODE_CLASS_MAPPINGS as M5, NODE_DISPLAY_NAME_MAPPINGS as D5
from .connectivity_refine import NODE_CLASS_MAPPINGS as M6, NODE_DISPLAY_NAME_MAPPINGS as D6

NODE_CLASS_MAPPINGS = {**M1, **M2, **M3, **M4, **M5, **M6}
NODE_DISPLAY_NAME_MAPPINGS = {**D1, **D2, **D3, **D4, **D5, **D6}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
