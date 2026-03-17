from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Column, String, DateTime, Integer, ForeignKey, Text, UniqueConstraint
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

    # Smart home: link to Room entity and household
    room_id = Column(String(36), ForeignKey('rooms.id', ondelete='SET NULL', use_alter=True), nullable=True)
    household_id = Column(String(255), nullable=True, index=True)

    # Relationships
    settings_requests = relationship("SettingsRequest", back_populates="node", cascade="all, delete-orphan")
    settings_snapshots = relationship("SettingsSnapshot", back_populates="node", cascade="all, delete-orphan")
    room_ref = relationship("Room", back_populates="nodes", foreign_keys=[room_id])
    config_pushes = relationship("ConfigPush", back_populates="node", cascade="all, delete-orphan")


class Room(Base):
    __tablename__ = 'rooms'
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    household_id = Column(String(255), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    normalized_name = Column(String(255), nullable=False)
    icon = Column(String(50), nullable=True)
    ha_area_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('household_id', 'normalized_name', name='uq_room_household_name'),
    )

    # Relationships
    devices = relationship("Device", back_populates="room")
    nodes = relationship("Node", back_populates="room_ref", foreign_keys=[Node.room_id])


class Device(Base):
    __tablename__ = 'devices'
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    household_id = Column(String(255), nullable=False, index=True)
    room_id = Column(String(36), ForeignKey('rooms.id', ondelete='SET NULL'), nullable=True)
    entity_id = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    domain = Column(String(50), nullable=False)
    device_class = Column(String(100), nullable=True)
    manufacturer = Column(String(255), nullable=True)
    model = Column(String(255), nullable=True)
    source = Column(String(50), default="home_assistant")
    protocol = Column(String(50), nullable=True)  # e.g., "lifx", "kasa", "tuya"
    local_ip = Column(String(45), nullable=True)   # IPv4/IPv6 address on LAN
    mac_address = Column(String(17), nullable=True)  # MAC for stable identity
    ha_device_id = Column(String(255), nullable=True)
    is_controllable = Column(Boolean, default=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('household_id', 'entity_id', name='uq_device_household_entity'),
    )

    # Relationships
    room = relationship("Room", back_populates="devices")


class ConfigPush(Base):
    """Encrypted config blob staged for a node to pick up via MQTT + poll.

    Flow:
    1. Mobile encrypts config with K2 and POSTs to CC
    2. CC stores the blob and publishes MQTT notification
    3. Node hears MQTT, GETs the blob from CC
    4. Node decrypts with K2 and writes to local secrets DB
    5. Node ACKs (marks consumed)

    Auth pushes (config_type="auth:*") set expires_at to auto-expire
    after 5 minutes, preventing stale tokens from sitting in the DB.
    """
    __tablename__ = 'config_pushes'
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    node_id = Column(String, ForeignKey('nodes.node_id', ondelete='CASCADE'), nullable=False)
    config_type = Column(String(50), nullable=False)  # e.g. "home_assistant", "auth:home_assistant"
    ciphertext = Column(Text, nullable=False)  # base64url encoded
    nonce = Column(Text, nullable=False)       # base64url encoded
    tag = Column(Text, nullable=False)         # base64url encoded
    status = Column(String(20), nullable=False, default="pending")  # pending, consumed
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    consumed_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)  # null = no expiry; auth pushes set 5 min TTL

    # Relationships
    node = relationship("Node", back_populates="config_pushes")


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


class AuthSession(Base):
    """OAuth session managed by JCC as redirect authority.

    Flow:
    1. Mobile creates session (status=pending) with provider + auth config
    2. JCC generates state + PKCE, builds authorize_url, returns session_id + URL
    3. Mobile opens authorize_url in WebView → provider redirects to JCC callback
    4. JCC validates state, exchanges code for tokens, encrypts + stores them
    5. Session marked active; MQTT notification sent
    6. Node pulls credentials from JCC via app-to-app auth

    Tokens are encrypted at rest with AES-256-GCM using JARVIS_TOKEN_ENCRYPTION_KEY.
    """
    __tablename__ = 'auth_sessions'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    provider = Column(String(100), nullable=False)           # "home_assistant"
    node_id = Column(String, ForeignKey('nodes.node_id', ondelete='CASCADE'), nullable=False)
    status = Column(String(20), nullable=False, default="pending")  # pending → active → consumed
    state = Column(String(64), nullable=False, unique=True)  # CSRF token
    code_verifier = Column(String(128), nullable=True)       # PKCE (null if provider doesn't support it)
    provider_base_url = Column(String(500), nullable=True)   # e.g., "http://192.168.1.100:8123"
    authorize_url = Column(Text, nullable=True)              # Full URL mobile opens
    exchange_url = Column(Text, nullable=True)               # Token exchange endpoint
    client_id = Column(String(255), nullable=False)
    redirect_uri = Column(Text, nullable=True)              # Redirect URI used (needed for code exchange)

    # Result (populated after successful exchange, encrypted at rest)
    access_token_enc = Column(Text, nullable=True)
    refresh_token_enc = Column(Text, nullable=True)
    token_data_enc = Column(Text, nullable=True)             # Full token response, encrypted

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)            # Session expires after 10 min
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    node = relationship("Node", backref="auth_sessions", passive_deletes=True)


class UserMemory(Base):
    """Persistent user memory for voice-identified personalization.

    Stores facts, preferences, and notes about users so Jarvis can
    personalize responses across conversations.
    """
    __tablename__ = 'user_memories'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False)
    household_id = Column(String(255), nullable=False)
    category = Column(String(100), nullable=False, default='general')  # preference, fact, note
    key = Column(String(255), nullable=True)  # optional structured key for upsert
    content = Column(Text, nullable=False)
    source = Column(String(50), nullable=False, default='voice')  # voice, ui, system
    is_active = Column(Boolean, nullable=False, default=True)
    is_pinned = Column(Boolean, nullable=False, server_default='false')
    embedding = Column(Vector(384), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)


class ProvisioningToken(Base):
    """Short-lived token for node self-registration.

    Flow:
    1. Authenticated user requests a provisioning token (gets GUID + raw token)
    2. Raw token is passed to the node (e.g., via mobile app)
    3. Node calls /nodes/register with node_id + raw token
    4. CC validates hash, creates node, consumes token
    """
    __tablename__ = "provisioning_tokens"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    token_hash = Column(String(64), nullable=False, unique=True)  # SHA-256 hex
    node_id = Column(String(36), nullable=False, index=True)
    household_id = Column(String(255), nullable=False)
    room = Column(String(255), nullable=True)
    name = Column(String(255), nullable=True)
    created_by_user_id = Column(Integer, nullable=True)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    consumed_at = Column(DateTime, nullable=True)
