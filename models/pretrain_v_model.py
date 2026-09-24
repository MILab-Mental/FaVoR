"""Compatibility imports for Video JEPA pretraining.

Video JEPA owns its model construction and checkpoint utilities. Keep this
module so existing callers can continue to import it during the transition.
"""

from models.video_jepa import init_video_model, load_checkpoint

__all__ = ["init_video_model", "load_checkpoint"]
