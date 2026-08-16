import time
import logging
from typing import Dict, Optional, List
from threading import Lock

logger = logging.getLogger("uvicorn")

# Type alias for conversation messages
ConversationMessage = Dict[str, str]


def _expects_continuation(msg: ConversationMessage) -> bool:
    """True if ``msg`` leaves its exchange open — the next message(s) belong to
    the SAME turn: a ``role="tool"`` result (more results or the final assistant
    reply follow), or an assistant message that issued ``tool_calls`` (its
    results — native ``role="tool"`` messages or the text-mode user-message
    injection — haven't landed yet)."""
    if not isinstance(msg, dict):
        return False
    role = msg.get("role")
    if role == "tool":
        return True
    return role == "assistant" and bool(msg.get("tool_calls"))


def trim_history_to_max_turns(
    messages: List[ConversationMessage],
    max_turns: int,
) -> int:
    """Sliding-window trim of conversation history, IN PLACE.

    Keeps the leading system prefix (the cached prompt at ``messages[0]`` plus
    any system messages before the first exchange) and only the most recent
    ``max_turns`` exchanges. A turn starts at a ``role="user"`` message and
    carries its whole exchange with it — the assistant reply, any assistant
    ``tool_calls``, the ``role="tool"`` results, and text-mode tool-result user
    injections — so a tool response is never orphaned and an assistant
    tool_call never loses its results (turns are kept or dropped atomically).

    Whole turns are dropped from the FRONT (oldest first) and retained messages
    are never reordered or rewritten, so the prompt stays byte-stable from the
    system prefix through the retained tail — the llama.cpp KV prefix cache
    only re-prefills from the trim boundary, never the (large) system prompt.

    ``max_turns <= 0`` disables trimming. Returns the number of messages
    dropped (0 = no trim).
    """
    if max_turns <= 0:
        return 0

    # Leading system prefix (the cached prompt) — never trimmed.
    prefix_end = 0
    while prefix_end < len(messages) and messages[prefix_end].get("role") == "system":
        prefix_end += 1

    # Group everything after the prefix into turns. A user message starts a
    # new turn unless the previous message still expects a continuation (tool
    # result pending / text-mode tool-result injection).
    turns: List[List[ConversationMessage]] = []
    for msg in messages[prefix_end:]:
        if msg.get("role") == "user" and (
            not turns or not _expects_continuation(turns[-1][-1])
        ):
            turns.append([msg])
        elif turns:
            turns[-1].append(msg)
        else:
            # Defensive: stray non-user message before the first exchange —
            # anchor a turn on it so it ages out with the window.
            turns.append([msg])

    if len(turns) <= max_turns:
        return 0

    kept = turns[len(turns) - max_turns:]
    trimmed = messages[:prefix_end] + [m for turn in kept for m in turn]
    dropped = len(messages) - len(trimmed)
    messages[:] = trimmed
    logger.info(
        f"✂️ Trimmed conversation history: dropped {len(turns) - max_turns} "
        f"turns ({dropped} messages), kept last {max_turns} turns"
    )
    return dropped


class ConversationCache:
    """Simple in-memory cache for conversation messages, tools, and available commands with TTL expiration."""
    
    def __init__(self, ttl_minutes: int = 10):
        self.cache: Dict[str, Dict] = {}
        self.ttl_seconds = ttl_minutes * 60
        self.lock = Lock()
        logger.info(f"🔧 ConversationCache initialized with {ttl_minutes} minute TTL")
    
    def set(
        self,
        conversation_id: str,
        messages: List[ConversationMessage],
        available_commands: List[Dict],
        timezone: str = None,
        tools: List[Dict] = None,
        node_context: Dict = None
    ) -> None:
        """Store messages, available commands, and tools for a conversation ID."""
        with self.lock:
            self.cache[conversation_id] = {
                'messages': messages,
                'available_commands': available_commands,
                'tools': tools or [],
                'timezone': timezone,
                'node_context': node_context,
                'router_decision': None,
                'referenced_items': None,
                'timestamp': time.time()
            }
            logger.info(f"💾 Cached messages, commands, and {len(tools or [])} tools for conversation {conversation_id[:8]}...")
    
    def get_messages(self, conversation_id: str) -> Optional[List[ConversationMessage]]:
        """Retrieve messages for a conversation ID if not expired."""
        with self.lock:
            if conversation_id not in self.cache:
                logger.debug(f"❌ No cached messages found for conversation {conversation_id[:8]}...")
                return None
            
            entry = self.cache[conversation_id]
            age_seconds = time.time() - entry['timestamp']
            
            if age_seconds > self.ttl_seconds:
                # Expired, remove it
                del self.cache[conversation_id]
                logger.info(f"⏰ Expired cache for conversation {conversation_id[:8]}... (age: {age_seconds:.1f}s)")
                return None
            
            logger.info(f"📖 Retrieved cached messages for conversation {conversation_id[:8]}... (age: {age_seconds:.1f}s)")
            return entry['messages']
    
    def get_available_commands(self, conversation_id: str) -> Optional[List[Dict]]:
        """Retrieve available commands for a conversation ID if not expired."""
        with self.lock:
            if conversation_id not in self.cache:
                logger.debug(f"❌ No cached available commands found for conversation {conversation_id[:8]}...")
                return None
            
            entry = self.cache[conversation_id]
            age_seconds = time.time() - entry['timestamp']
            
            if age_seconds > self.ttl_seconds:
                # Expired, remove it
                del self.cache[conversation_id]
                logger.info(f"⏰ Expired cache for conversation {conversation_id[:8]}... (age: {age_seconds:.1f}s)")
                return None
            
            logger.info(f"📖 Retrieved cached available commands for conversation {conversation_id[:8]}... (age: {age_seconds:.1f}s)")
            return entry['available_commands']
    
    def get_timezone(self, conversation_id: str) -> Optional[str]:
        """Retrieve timezone for a conversation ID if not expired."""
        with self.lock:
            if conversation_id not in self.cache:
                logger.debug(f"❌ No cached timezone found for conversation {conversation_id[:8]}...")
                return None
            
            entry = self.cache[conversation_id]
            age_seconds = time.time() - entry['timestamp']
            
            if age_seconds > self.ttl_seconds:
                # Expired, remove it
                del self.cache[conversation_id]
                logger.info(f"⏰ Expired cache for conversation {conversation_id[:8]}... (age: {age_seconds:.1f}s)")
                return None
            
            logger.info(f"📖 Retrieved cached timezone for conversation {conversation_id[:8]}... (age: {age_seconds:.1f}s)")
            return entry.get('timezone')

    def get_node_context(self, conversation_id: str) -> Optional[Dict]:
        """Retrieve node_context for a conversation ID if not expired."""
        with self.lock:
            if conversation_id not in self.cache:
                logger.debug(f"❌ No cached node_context found for conversation {conversation_id[:8]}...")
                return None

            entry = self.cache[conversation_id]
            age_seconds = time.time() - entry['timestamp']

            if age_seconds > self.ttl_seconds:
                del self.cache[conversation_id]
                logger.info(f"⏰ Expired cache for conversation {conversation_id[:8]}... (age: {age_seconds:.1f}s)")
                return None

            logger.info(f"📖 Retrieved cached node_context for conversation {conversation_id[:8]}... (age: {age_seconds:.1f}s)")
            return entry.get('node_context')

    def set_router_decision(self, conversation_id: str, decision: Optional[Dict]) -> None:
        """Store the latest router decision for a conversation."""
        with self.lock:
            if conversation_id not in self.cache:
                logger.warning(f"❌ Cannot set router decision for non-existent conversation {conversation_id[:8]}...")
                return
            entry = self.cache[conversation_id]
            age_seconds = time.time() - entry['timestamp']
            if age_seconds > self.ttl_seconds:
                del self.cache[conversation_id]
                logger.warning(f"❌ Cannot set router decision for expired conversation {conversation_id[:8]}...")
                return
            entry['router_decision'] = decision
            logger.info(f"🧭 Stored router decision for conversation {conversation_id[:8]}...")

    def get_router_decision(self, conversation_id: str) -> Optional[Dict]:
        """Retrieve router decision for a conversation ID if not expired."""
        with self.lock:
            if conversation_id not in self.cache:
                logger.debug(f"❌ No cached router decision found for conversation {conversation_id[:8]}...")
                return None
            entry = self.cache[conversation_id]
            age_seconds = time.time() - entry['timestamp']
            if age_seconds > self.ttl_seconds:
                del self.cache[conversation_id]
                logger.info(f"⏰ Expired cache for conversation {conversation_id[:8]}... (age: {age_seconds:.1f}s)")
                return None
            return entry.get('router_decision')

    def set_wake_verification(self, conversation_id: str, verdict: Optional[Dict]) -> None:
        """Store the wake-clip verification verdict for a conversation.

        Written by the media proxy's background verify task; read by the
        command turn with a short bounded wait (see app/core/wake_verification).
        A missing entry always means "fail open" downstream."""
        with self.lock:
            if conversation_id not in self.cache:
                logger.warning(f"❌ Cannot set wake verification for non-existent conversation {conversation_id[:8]}...")
                return
            entry = self.cache[conversation_id]
            age_seconds = time.time() - entry['timestamp']
            if age_seconds > self.ttl_seconds:
                del self.cache[conversation_id]
                logger.warning(f"❌ Cannot set wake verification for expired conversation {conversation_id[:8]}...")
                return
            entry['wake_verification'] = verdict
            logger.info(f"🔎 Stored wake verification for conversation {conversation_id[:8]}...")

    def get_wake_verification(self, conversation_id: str) -> Optional[Dict]:
        """Retrieve the wake-clip verdict (None = absent/expired → fail open)."""
        with self.lock:
            if conversation_id not in self.cache:
                return None
            entry = self.cache[conversation_id]
            age_seconds = time.time() - entry['timestamp']
            if age_seconds > self.ttl_seconds:
                del self.cache[conversation_id]
                return None
            return entry.get('wake_verification')

    def set_referenced_items(self, conversation_id: str, items: Optional[List[Dict]]) -> None:
        """Store the items a tool just surfaced ("recently shown" list).

        These are re-injected into the prompt each turn so the LLM can resolve
        "those" / "#3" / "the one from abc" to a ref_id and call act_on_items.
        Latest surfacing wins; callers pass None/[] only when there is something
        to store (a non-surfacing turn leaves the prior list intact)."""
        with self.lock:
            if conversation_id not in self.cache:
                logger.warning(f"❌ Cannot set referenced items for non-existent conversation {conversation_id[:8]}...")
                return
            entry = self.cache[conversation_id]
            age_seconds = time.time() - entry['timestamp']
            if age_seconds > self.ttl_seconds:
                del self.cache[conversation_id]
                logger.warning(f"❌ Cannot set referenced items for expired conversation {conversation_id[:8]}...")
                return
            entry['referenced_items'] = items
            logger.info(f"🗂️ Stored {len(items or [])} referenced items for conversation {conversation_id[:8]}...")

    def get_referenced_items(self, conversation_id: str) -> List[Dict]:
        """Retrieve the recently-shown items for a conversation (empty if none/expired)."""
        with self.lock:
            if conversation_id not in self.cache:
                return []
            entry = self.cache[conversation_id]
            age_seconds = time.time() - entry['timestamp']
            if age_seconds > self.ttl_seconds:
                del self.cache[conversation_id]
                return []
            return entry.get('referenced_items') or []


    def add_message(self, conversation_id: str, message: ConversationMessage) -> None:
        """Add a message to the existing conversation."""
        with self.lock:
            if conversation_id not in self.cache:
                logger.warning(f"❌ Cannot add message to non-existent conversation {conversation_id[:8]}...")
                return
            
            entry = self.cache[conversation_id]
            age_seconds = time.time() - entry['timestamp']
            
            if age_seconds > self.ttl_seconds:
                # Expired, remove it
                del self.cache[conversation_id]
                logger.warning(f"❌ Cannot add message to expired conversation {conversation_id[:8]}...")
                return
            
            entry['messages'].append(message)
            logger.info(f"📝 Added message to conversation {conversation_id[:8]}... (total messages: {len(entry['messages'])})")
    
    def add_messages(self, conversation_id: str, messages: List[ConversationMessage]) -> None:
        """Add multiple messages to the existing conversation."""
        with self.lock:
            if conversation_id not in self.cache:
                logger.warning(f"❌ Cannot add messages to non-existent conversation {conversation_id[:8]}...")
                return
            
            entry = self.cache[conversation_id]
            age_seconds = time.time() - entry['timestamp']
            
            if age_seconds > self.ttl_seconds:
                # Expired, remove it
                del self.cache[conversation_id]
                logger.warning(f"❌ Cannot add messages to expired conversation {conversation_id[:8]}...")
                return
            
            entry['messages'].extend(messages)
            logger.info(f"📝 Added {len(messages)} messages to conversation {conversation_id[:8]}... (total messages: {len(entry['messages'])})")
    
    def update_messages(self, conversation_id: str, messages: List[ConversationMessage]) -> None:
        """Replace all messages in the conversation."""
        with self.lock:
            if conversation_id not in self.cache:
                logger.warning(f"❌ Cannot update messages for non-existent conversation {conversation_id[:8]}...")
                return
            
            entry = self.cache[conversation_id]
            age_seconds = time.time() - entry['timestamp']
            
            if age_seconds > self.ttl_seconds:
                # Expired, remove it
                del self.cache[conversation_id]
                logger.warning(f"❌ Cannot update messages for expired conversation {conversation_id[:8]}...")
                return
            
            entry['messages'] = messages
            logger.info(f"📝 Updated messages for conversation {conversation_id[:8]}... (total messages: {len(messages)})")
    
    def get_tools(self, conversation_id: str) -> Optional[List[Dict]]:
        """Retrieve tools for a conversation ID if not expired."""
        with self.lock:
            if conversation_id not in self.cache:
                logger.debug(f"❌ No cached tools found for conversation {conversation_id[:8]}...")
                return None
            
            entry = self.cache[conversation_id]
            age_seconds = time.time() - entry['timestamp']
            
            if age_seconds > self.ttl_seconds:
                # Expired, remove it
                del self.cache[conversation_id]
                logger.info(f"⏰ Expired cache for conversation {conversation_id[:8]}... (age: {age_seconds:.1f}s)")
                return None
            
            logger.info(f"📖 Retrieved cached tools for conversation {conversation_id[:8]}... (age: {age_seconds:.1f}s)")
            return entry.get('tools', [])
    
    def remove(self, conversation_id: str) -> None:
        """Remove a specific conversation from the cache."""
        with self.lock:
            if conversation_id in self.cache:
                del self.cache[conversation_id]
                logger.info(f"🗑️ Removed cached messages for conversation {conversation_id[:8]}...")
    
    def clear_expired(self) -> int:
        """Remove all expired entries and return count of removed items."""
        current_time = time.time()
        expired_keys = []
        
        with self.lock:
            for conv_id, entry in self.cache.items():
                if current_time - entry['timestamp'] > self.ttl_seconds:
                    expired_keys.append(conv_id)
            
            for conv_id in expired_keys:
                del self.cache[conv_id]
        
        if expired_keys:
            logger.info(f"🧹 Cleaned up {len(expired_keys)} expired conversation cache entries")
        
        return len(expired_keys)
    
    def size(self) -> int:
        """Get current cache size."""
        with self.lock:
            return len(self.cache)
    
    def stats(self) -> Dict:
        """Get cache statistics."""
        current_time = time.time()
        with self.lock:
            active_entries = 0
            expired_entries = 0
            
            for entry in self.cache.values():
                if current_time - entry['timestamp'] <= self.ttl_seconds:
                    active_entries += 1
                else:
                    expired_entries += 1
            
            return {
                'total_entries': len(self.cache),
                'active_entries': active_entries,
                'expired_entries': expired_entries,
                'ttl_seconds': self.ttl_seconds
            }
    
    def set_force_tool_calls(self, conversation_id: str, force: bool) -> None:
        """Store whether tool calls should be unconditionally enforced."""
        with self.lock:
            if conversation_id not in self.cache:
                logger.warning(f"❌ Cannot set force_tool_calls for non-existent conversation {conversation_id[:8]}...")
                return
            entry = self.cache[conversation_id]
            age_seconds = time.time() - entry['timestamp']
            if age_seconds > self.ttl_seconds:
                del self.cache[conversation_id]
                return
            entry['force_tool_calls'] = force

    def get_force_tool_calls(self, conversation_id: str) -> bool:
        """Check whether tool calls should be unconditionally enforced."""
        with self.lock:
            if conversation_id not in self.cache:
                return False
            entry = self.cache[conversation_id]
            age_seconds = time.time() - entry['timestamp']
            if age_seconds > self.ttl_seconds:
                del self.cache[conversation_id]
                return False
            return entry.get('force_tool_calls', False)

    def has_conversation(self, conversation_id: str) -> bool:
        """Check if a conversation exists in the cache and is not expired."""
        with self.lock:
            if conversation_id not in self.cache:
                return False
            
            entry = self.cache[conversation_id]
            age_seconds = time.time() - entry['timestamp']
            
            if age_seconds > self.ttl_seconds:
                # Expired, remove it
                del self.cache[conversation_id]
                return False
            
            return True

# Global instance
conversation_cache = ConversationCache()
