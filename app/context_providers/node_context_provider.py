from app.core.interfaces.ijarvis_context_provider import IJarvisContextProvider
from app.models import Node
from typing import Dict
import json


class NodeContextProvider(IJarvisContextProvider):
    key: str = "NodeContext"
    node: Node
    household_id: str | None

    def __init__(self, node: Node, household_id: str | None = None):
        self.node = node
        self.household_id = household_id


    def get_context(self):
        return ""

    @property
    def context_summary(self):
        return f"The user's command came from a microphone with this context: {json.dumps({'room': self.node.room, 'node_id': self.node.node_id})}"

