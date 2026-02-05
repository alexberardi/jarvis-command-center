from sqlalchemy import Boolean, Column, String, DateTime, Integer, ForeignKey, Text, UniqueConstraint, func
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime, timedelta

Base = declarative_base()


def _default_expires_at() -> datetime:
    """Default expiration: 5 minutes from now."""
    return datetime.utcnow() + timedelta(minutes=5)


class Node(Base):
    __tablename__ = 'nodes'
    node_id = Column(String, primary_key=True)
    api_key = Column(String, nullable=False)
    room = Column(String, nullable=False)
    user = Column(String, default="default")
    voice_mode = Column(String, default="brief")
    last_seen = Column(DateTime, default=datetime.utcnow)
    # Per-node LoRA adapter hash (set after training completes)
    adapter_hash = Column(String, nullable=True)

    # Relationships
    settings_requests = relationship("SettingsRequest", back_populates="node", cascade="all, delete-orphan")
    settings_snapshots = relationship("SettingsSnapshot", back_populates="node", cascade="all, delete-orphan")


class SettingsRequest(Base):
    """
    A request from mobile to retrieve node settings.

    Lifecycle:
    1. Mobile creates request (status=pending)
    2. CC publishes MQTT signal to node
    3. Node confirms request via GET
    4. Node uploads encrypted snapshot (status=fulfilled)
    5. Mobile retrieves snapshot

    Requests expire after expires_at to prevent replay attacks.
    """
    __tablename__ = 'settings_requests'

    request_id = Column(String, primary_key=True)
    node_id = Column(String, ForeignKey('nodes.node_id', ondelete='CASCADE'), nullable=False)
    status = Column(String, nullable=False, default="pending")  # pending, fulfilled, expired
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False, default=_default_expires_at)

    # Relationships
    node = relationship("Node", back_populates="settings_requests")
    snapshot = relationship("SettingsSnapshot", back_populates="request", uselist=False)


class SettingsSnapshot(Base):
    """
    An encrypted settings snapshot uploaded by a node.

    The ciphertext is encrypted with K2 (mobile↔node key) using AES-256-GCM.
    AAD fields are stored separately so clients can reconstruct the AAD for decryption.

    AAD binding prevents ciphertext substitution attacks across nodes/requests.
    """
    __tablename__ = 'settings_snapshots'

    snapshot_id = Column(String, primary_key=True)
    node_id = Column(String, ForeignKey('nodes.node_id', ondelete='CASCADE'), nullable=False)
    request_id = Column(String, ForeignKey('settings_requests.request_id', ondelete='CASCADE'), nullable=False)

    # Encrypted payload (AES-256-GCM)
    ciphertext = Column(String, nullable=False)  # base64url encoded
    nonce = Column(String, nullable=False)       # base64url encoded (IV)
    tag = Column(String, nullable=False)         # base64url encoded (auth tag)

    # AAD components - stored separately for client-side AAD reconstruction
    # Clients must use these exact values when constructing AAD for decryption
    aad_node_id = Column(String, nullable=False)
    aad_schema_version = Column(Integer, nullable=False)
    aad_commands_schema_version = Column(Integer, nullable=False)
    aad_revision = Column(Integer, nullable=False)
    aad_request_id = Column(String, nullable=False)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    node = relationship("Node", back_populates="settings_snapshots")
    request = relationship("SettingsRequest", back_populates="snapshot")


class Setting(Base):
    """
    Scoped settings table with cascade lookup: User > Node > Household > Default.

    Settings can be defined at four levels:
    - System default: all scope fields NULL
    - Household-wide: household_id set, node_id=NULL, user_id=NULL
    - Node-specific: household_id and node_id set, user_id=NULL
    - User-specific: all scope fields set

    The SettingsService handles cascade lookup to find the most specific
    value for a given key.
    """
    __tablename__ = 'settings'

    id = Column(Integer, primary_key=True)
    key = Column(String(255), nullable=False, index=True)
    value = Column(Text, nullable=True)  # JSON-encoded
    value_type = Column(String(50), nullable=False, default="string")  # string, int, float, bool, json
    category = Column(String(100), nullable=False, default="general", index=True)
    description = Column(Text, nullable=True)
    requires_reload = Column(Boolean, default=False)
    is_secret = Column(Boolean, default=False)
    env_fallback = Column(String(255), nullable=True)

    # Multi-tenant scoping
    household_id = Column(String(255), nullable=True, index=True)  # NULL = system default
    node_id = Column(String(255), nullable=True, index=True)       # NULL = household-wide
    user_id = Column(Integer, nullable=True, index=True)           # NULL = not user-specific

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('key', 'household_id', 'node_id', 'user_id', name='uq_setting_scope'),
    )
