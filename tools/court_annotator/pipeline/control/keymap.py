import logging
from typing import Dict, List


class KeyMapper:
    """
    Resolves raw key names to abstract commands using cfg ui.binds as the single source of truth.
    Performs collision detection and logs warnings.
    """

    def __init__(self, binds: Dict[str, List[str]]):
        # Normalize binds: command -> list[str]
        self.binds = {cmd.lower(): [k.lower() for k in keys] for cmd, keys in binds.items()}
        self.key_to_cmd: Dict[str, str] = {}
        self.collisions: Dict[str, List[str]] = {}
        self._build_reverse()

    def _build_reverse(self):
        for cmd, keys in self.binds.items():
            for k in keys:
                if k in self.key_to_cmd and self.key_to_cmd[k] != cmd:
                    # record collision
                    self.collisions.setdefault(k, []).append(self.key_to_cmd[k])
                    self.collisions[k].append(cmd)
                self.key_to_cmd[k] = cmd
        for k, cmds in self.collisions.items():
            uniq = list(dict.fromkeys(cmds))
            logging.warning(f"Key bind collision on '{k}': mapped to {uniq}")

    def map_keys(self, raw_events: List[dict]) -> List[dict]:
        commands = []
        for ev in raw_events:
            if ev.get("type") != "key":
                continue
            name = ev.get("key", "").lower()
            cmd = self.key_to_cmd.get(name)
            if cmd:
                commands.append({"cmd": cmd})
        return commands
