from .runner import Stage
from ..domain.state import FrameState


class StateFetchStage(Stage):
    """
    A pipeline stage to fetch the state for the current frame from the
    session, or initialize it if it doesn't exist.
    """

    name = "state_fetch"

    def process(self, bundle):
        """
        Retrieves the FrameState for the current frame_idx from the
        session_state. If it's not found, a new FrameState is created.
        """
        frame_idx = bundle["frame_idx"]
        session_state = bundle["session_state"]

        if frame_idx not in session_state["frame_states"]:
            session_state["frame_states"][frame_idx] = FrameState()

        bundle["frame_state"] = session_state["frame_states"][frame_idx]
        return bundle
