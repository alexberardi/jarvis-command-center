from app.core.interfaces.ijarvis_context_provider import IJarvisContextProvider
from app.models import Node
import json


class NodeContextProvider(IJarvisContextProvider):
    key: str = "NodeContext"
    node: Node
    household_id: str | None
    household_member_ids: list[int]

    def __init__(
        self,
        node: Node,
        household_id: str | None = None,
        household_member_ids: list[int] | None = None,
    ):
        self.node = node
        self.household_id = household_id
        self.household_member_ids = household_member_ids or []


    def get_context(self):
        return ""

    @property
    def context_summary(self):
        return f"The user's command came from a microphone with this context: {json.dumps({'room': self.node.room, 'node_id': self.node.node_id})}"

