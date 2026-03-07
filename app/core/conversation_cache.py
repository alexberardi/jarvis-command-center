import time
import logging
from typing import Dict, Optional, List
from threading import Lock

logger = logging.getLogger("uvicorn")

# Type alias for conversation messages
ConversationMessage = Dict[str, str]

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
