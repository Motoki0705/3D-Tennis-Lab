from .runner import Stage


class PreprocessStage(Stage):
    name = "preprocess"

    def process(self, bundle):
        # This will determine display scale, etc.
        return bundle
