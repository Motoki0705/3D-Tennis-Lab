from .runner import Stage


class SourceStage(Stage):
    name = "source"

    def process(self, bundle):
        """Inject immutable video properties such as fps into the bundle."""
        video = bundle.get("video")
        if video is not None:
            bundle["fps"] = getattr(video, "fps", bundle.get("fps"))
            # Keep nframes consistent if not already set by the loop
            bundle["nframes"] = bundle.get("nframes", getattr(video, "nframes", 0))
        return bundle
