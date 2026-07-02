"""Trust / owner authorization manager.

Wraps the knowledge base to provide a simple API for ownership and threat
levels. Used by the control panel and in-game command handlers.
"""

from typing import Optional

from .knowledge_base import KnowledgeBase


class TrustManager:
    """Manages who the bot obeys and who it treats as hostile."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def set_owner(self, uuid: str, name: str) -> None:
        self.kb.set_owner(uuid, name)

    def trust_player(self, uuid: str, name: str) -> None:
        self.kb.set_threat(uuid, name, "ally")

    def neutral_player(self, uuid: str, name: str) -> None:
        self.kb.set_threat(uuid, name, "neutral")

    def mark_hostile(self, uuid: str, name: str) -> None:
        self.kb.set_threat(uuid, name, "hostile")

    def is_owner(self, uuid_or_name: str) -> bool:
        return self.kb.is_owner(uuid_or_name)

    def is_authorized(self, uuid_or_name: str) -> bool:
        """Owner and allies may issue commands."""
        row = self.kb.get_player(uuid_or_name) or self.kb.get_player_by_name(uuid_or_name)
        if not row:
            return False
        return row["threat_level"] in ("owner", "ally") or bool(row["owner"])

    def is_hostile(self, uuid_or_name: str) -> bool:
        return self.kb.is_hostile(uuid_or_name)

    def handle_chat(self, chat_event: dict) -> Optional[str]:
        """Process an owner/ally chat command. Returns response text or None."""
        if not chat_event:
            return None
        username = chat_event.get("username")
        message = (chat_event.get("message") or "").strip()
        if not username or not message:
            return None

        if not self.is_authorized(username):
            return None

        parts = message.split()
        if len(parts) < 2 or parts[0].lower() != "harvy":
            return None

        cmd = parts[1].lower()
        args = parts[2:]

        if cmd == "trust" and args:
            self.trust_player("", args[0])
            return f"{args[0]} is now trusted."
        if cmd == "hostile" and args:
            self.mark_hostile("", args[0])
            return f"{args[0]} is now hostile."
        if cmd == "neutral" and args:
            self.neutral_player("", args[0])
            return f"{args[0]} is now neutral."

        return None
