from .runner import Stage
from ..ui.draw import render as render_frame


class RenderStage(Stage):
    """A pipeline stage for rendering the output frame."""

    name = "render"

    def process(self, bundle):
        """
        Renders all visual elements (points, skeleton, HUD) onto the frame.
        The result is stored in bundle["rendered_frame"].
        """
        # The main rendering logic is delegated to the ui.draw module
        bundle = render_frame(bundle)
        return bundle
