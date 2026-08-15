from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Column, Float, String, DateTime, Integer, ForeignKey, Text, UniqueConstraint
from sqlalchemy.orm import backref, declarative_base, relationship
from datetime import datetime, timedelta

Base = declarative_base()


def _default_expires_at() -> datetime:
    """Default expiration: 5 minutes from now."""
    return datetime.utcnow() + timedelta(minutes=5)


ONLINE_THRESHOLD_MINUTES = 15


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

    # Reported via heartbeat; drives the mobile "update available" badge and
    # the busy-lock that defers update dispatch while the node is in session.
    last_seen_version = Column(String(64), nullable=True)
    install_mode = Column(String(16), nullable=True)  # "tarball" | "docker" | "dev"
    git_sha = Column(String(40), nullable=True)
    is_busy = Column(Boolean, default=False, nullable=False)

    # Soft-delete flag. False after a successful factory reset — the row
    # is kept so node_tasks FKs remain valid, but list queries should
    # filter on is_active=True.
    is_active = Column(Boolean, default=True, nullable=False)

    # JSON list of protocol names installed on this node, e.g. '["homeconnect","govee"]'.
    # Populated from heartbeat. Used to route device commands to a node that
    # has the required protocol adapter.
    protocols = Column(Text, nullable=True)

    # Reported via heartbeat: True when the node has no K2 secrets-encryption
    # key. Mobile uses this to show the settings gear so the user can pair K2
    # — without it, container nodes (which never go through AP provisioning)
    # are stuck with no way to add K2.
    needs_k2 = Column(Boolean, default=True, nullable=False)

    def is_online(self) -> bool:
        """Node is online if last_seen is within ONLINE_THRESHOLD_MINUTES."""
        if self.last_seen is None:
            return False
        cutoff = datetime.utcnow() - timedelta(minutes=ONLINE_THRESHOLD_MINUTES)
        return self.last_seen >= cutoff

    # Relationships
    settings_requests = relationship("SettingsRequest", back_populates="node", cascade="all, delete-orphan")
    settings_snapshots = relationship("SettingsSnapshot", back_populates="node", cascade="all, delete-orphan")
    room_ref = relationship("Room", back_populates="nodes", foreign_keys=[room_id])
    config_pushes = relationship("ConfigPush", back_populates="node", cascade="all, delete-orphan")
    tasks = relationship("NodeTask", back_populates="node", cascade="all, delete-orphan")


class NodeTask(Base):
    """Long-running per-node operations (update, reconfig, etc.).

    State machine: pending → dispatched → in_progress → success | failed.
    The heartbeat handler dispatches pending tasks and the node reports
    progress by re-heartbeating with a new version_info.
    """
    __tablename__ = 'node_tasks'
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    node_id = Column(String, ForeignKey('nodes.node_id', ondelete='CASCADE'), nullable=False, index=True)
    kind = Column(String(32), nullable=False)  # e.g. "update"
    target_version = Column(String(64), nullable=True)
    state = Column(String(32), nullable=False, default="pending")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    finished_at = Column(DateTime, nullable=True)

    node = relationship("Node", back_populates="tasks")


class Room(Base):
    __tablename__ = 'rooms'
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    household_id = Column(String(255), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    normalized_name = Column(String(255), nullable=False)
    icon = Column(String(50), nullable=True)
    ha_area_id = Column(String(255), nullable=True)
    parent_room_id = Column(String(36), ForeignKey('rooms.id', ondelete='SET NULL'), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('household_id', 'normalized_name', name='uq_room_household_name'),
    )

    # Relationships
    devices = relationship("Device", back_populates="room")
    nodes = relationship("Node", back_populates="room_ref", foreign_keys=[Node.room_id])
    children = relationship("Room", backref=backref("parent", remote_side="Room.id"), cascade="all")


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
    cloud_id = Column(String(255), nullable=True)     # Cloud-only device ID (Govee, Nest, Schlage)
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


class DeviceScanRequest(Base):
    """User-driven device scan request: mobile → CC → MQTT → node → CC → mobile.

    Lifecycle:
    1. Mobile POSTs to request a scan (status=pending, expires_at=now+2min)
    2. CC publishes MQTT to node's device-scan topic
    3. Node runs protocol adapters, POSTs results back
    4. Mobile polls for results (CC enriches with already_registered flags)
    """
    __tablename__ = 'device_scan_requests'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    node_id = Column(String, ForeignKey('nodes.node_id', ondelete='CASCADE'), nullable=False)
    household_id = Column(String(255), nullable=False, index=True)
    status = Column(String(20), nullable=False, default="pending")  # pending, completed, failed, expired
    results_json = Column(Text, nullable=True)  # JSON array of discovered devices
    device_count = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    node = relationship("Node", backref=backref("scan_requests", passive_deletes=True), passive_deletes=True)


class DeviceListRequest(Base):
    """Device list request: mobile → CC → MQTT → node → CC → mobile.

    Lifecycle:
    1. Mobile POSTs to request a device list (status=pending, expires_at=now+2min)
    2. CC publishes MQTT to node's device-list topic with selected manager_name
    3. Node runs the selected IJarvisDeviceManager, POSTs results back
    4. Mobile polls for results (CC enriches with room assignments)
    """
    __tablename__ = 'device_list_requests'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    node_id = Column(String, ForeignKey('nodes.node_id', ondelete='CASCADE'), nullable=False)
    household_id = Column(String(255), nullable=False, index=True)
    status = Column(String(20), nullable=False, default="pending")  # pending, completed, failed, expired
    manager_name = Column(String(100), nullable=True)  # which manager produced the list
    can_edit_devices = Column(Boolean, nullable=True)  # from the manager
    results_json = Column(Text, nullable=True)  # JSON array of device dicts
    device_count = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    node = relationship("Node", backref=backref("device_list_requests", passive_deletes=True), passive_deletes=True)


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
    client_secret_enc = Column(Text, nullable=True)         # Encrypted client_secret (Web Application OAuth)
    redirect_uri = Column(Text, nullable=True)              # Redirect URI used (needed for code exchange)
    user_id = Column(Integer, nullable=True)                # Initiating user — owner of user-scoped tokens

    # Result (populated after successful exchange, encrypted at rest)
    access_token_enc = Column(Text, nullable=True)
    refresh_token_enc = Column(Text, nullable=True)
    token_data_enc = Column(Text, nullable=True)             # Full token response, encrypted

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)            # Session expires after 10 min
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    node = relationship("Node", backref=backref("auth_sessions", passive_deletes=True), passive_deletes=True)


class UserMemory(Base):
    """Persistent user memory for voice-identified personalization.

    Stores facts, preferences, and notes about users so Jarvis can
    personalize responses across conversations.
    """
    __tablename__ = 'user_memories'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=True)  # NULL = household-wide (agent context)
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


class PersonCharacterization(Base):
    """Synthesized, evolving "view of a person" — one row per (user, household).

    A background job (``characterization_synthesis_service``) regenerates a single
    document from the person's facts + recent transcripts: a structured ``body``
    (JSON) plus a ``rendered`` prose summary that can be injected into the voice
    prompt as a ``<person_view>`` tail (gated, default OFF). Distinct from
    ``UserMemory`` (discrete facts) and the persona (household VOICE) — this is
    Jarvis's model of WHO it's talking to, and it only grows richer over time.
    """
    __tablename__ = 'person_characterizations'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False)
    household_id = Column(String(255), nullable=False)
    body = Column(Text, nullable=False)  # structured JSON document
    rendered = Column(Text, nullable=True)  # prose summary for prompt injection
    confidence = Column(Float, nullable=True)
    version = Column(Integer, nullable=False, server_default='1')
    last_transcript_at = Column(DateTime, nullable=True)  # newest transcript folded in
    model = Column(String(100), nullable=True)  # which background model synthesized it
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            'user_id', 'household_id', name='uq_person_characterization_user_household'
        ),
    )


class Signal(Base):
    """A generic, self-describing world-state observation for the Signal Bus.

    Any producer (node agent, phone geofence, an internal microservice, or an
    external system via the API) writes a Signal; the reactive renderer surfaces
    it to the LLM and the proactive matcher reasons over it. ``kind`` is a
    descriptive label, NEVER control flow. ``source_key`` is the stable dedup +
    suppression key: the same logical fact re-observed UPSERTs one row.

    Stored in its OWN table (not ``user_memories``) so situational/location
    facts never leak into the recall tool or the embedding sweep.
    """
    __tablename__ = 'signals'

    id = Column(Integer, primary_key=True, autoincrement=True)
    household_id = Column(String(255), nullable=False, index=True)
    user_id = Column(Integer, nullable=True, index=True)     # NULL = household-wide
    node_id = Column(String(255), nullable=True)
    room = Column(String(255), nullable=True)
    kind = Column(String(255), nullable=False, index=True)   # label only, never control flow
    subject = Column(String(255), nullable=True)             # sub-key within a kind
    source_key = Column(String(512), nullable=False)         # stable dedup + suppression key
    summary = Column(Text, nullable=True)                    # the NL line the matcher reads
    facts = Column(Text, nullable=True)                      # typed payload, JSON-serialized
    source_agent = Column(String(255), nullable=True)        # self-declared producer id
    cacheable = Column(Boolean, nullable=False, default=False)   # may ride the cached prefix?
    salience = Column(Float, nullable=True)                  # capped ranking hint
    observed_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True, index=True)  # NULL = never expires
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('household_id', 'source_key', name='uq_signals_household_source'),
    )


class ConversationTranscript(Base):
    """Buffered voice conversation transcript for passive memory extraction.

    Logged fire-and-forget after each voice interaction. A background task
    batches unprocessed transcripts per user, sends them to the LLM background
    model for memory extraction, then marks them processed. Expired rows are
    cleaned up based on a configurable TTL (default 7 days).
    """
    __tablename__ = 'conversation_transcripts'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, index=True)
    household_id = Column(String(255), nullable=False)
    conversation_id = Column(String(255), nullable=False)
    user_message = Column(Text, nullable=False)
    assistant_message = Column(Text, nullable=True)
    tool_calls_json = Column(Text, nullable=True)
    is_processed = Column(Boolean, nullable=False, default=False, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    processed_at = Column(DateTime, nullable=True)
    extraction_job_id = Column(String(36), nullable=True)

    # Phase 1 feedback: user rates the parsed command (−1 / 0 / +1).
    # Feeds the Phase 3 training-data extractor as explicit positive/negative signal.
    user_rating = Column(Integer, nullable=True)
    rating_notes = Column(Text, nullable=True)
    rated_at = Column(DateTime, nullable=True)


class PackageInstallRequest(Base):
    """Package install request: mobile -> CC -> MQTT -> node -> CC -> mobile.

    Lifecycle:
    1. Mobile POSTs to request a package install (status=pending, expires_at=now+5min)
    2. CC publishes MQTT to node's package-install topic with repo info
    3. Node clones repo, runs install pipeline, POSTs result back
    4. Mobile polls for results
    """
    __tablename__ = 'package_install_requests'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    node_id = Column(String, ForeignKey('nodes.node_id', ondelete='CASCADE'), nullable=False)
    household_id = Column(String(255), nullable=False, index=True)
    command_name = Column(String(255), nullable=False)
    github_repo_url = Column(Text, nullable=False)
    git_tag = Column(String(100), nullable=True)
    status = Column(String(20), nullable=False, default="pending")  # pending, restarting, completed, failed, expired
    results_json = Column(Text, nullable=True)  # JSON object with install result
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    node = relationship("Node", backref=backref("package_install_requests", passive_deletes=True), passive_deletes=True)


class TestInstallRequest(Base):
    """Test install request: mobile -> CC -> MQTT nudge -> node verifies -> downloads from Pantry.

    Lifecycle:
    1. Mobile POSTs share_code + node_id (status=pending, expires_at=now+5min)
    2. CC publishes MQTT with just request_id (lightweight nudge)
    3. Node verifies with CC, gets Pantry download URL, installs to test_commands/
    4. Node POSTs result back, mobile polls for status
    """
    __tablename__ = 'test_install_requests'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    node_id = Column(String, ForeignKey('nodes.node_id', ondelete='CASCADE'), nullable=False)
    household_id = Column(String(255), nullable=False, index=True)
    share_code = Column(String(6), nullable=False)
    package_name = Column(String(255), nullable=False)
    status = Column(String(20), nullable=False, default="pending")  # pending, completed, failed, expired
    results_json = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    node = relationship("Node", backref=backref("test_install_requests", passive_deletes=True), passive_deletes=True)


class PromptProviderInstallRequest(Base):
    """Async prompt provider install request: mobile -> CC -> background task -> mobile polls.

    Lifecycle:
    1. Mobile POSTs to request a prompt provider install (status=pending, expires_at=now+5min)
    2. CC kicks off BackgroundTasks to clone repo, validate, and install
    3. Background task updates status to completed/failed
    4. Mobile polls for results
    """
    __tablename__ = 'prompt_provider_install_requests'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    household_id = Column(String(255), nullable=False, index=True)
    package_name = Column(String(255), nullable=True)
    github_repo_url = Column(Text, nullable=False)
    git_tag = Column(String(100), nullable=True)
    status = Column(String(20), nullable=False, default="pending")  # pending, completed, failed, expired
    results_json = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)


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


class ActiveAdapter(Base):
    """Currently-deployed LoRA adapter for a household. One row per household."""
    __tablename__ = "active_adapter"

    household_id = Column(String(255), primary_key=True)
    adapter_hash = Column(String(128), nullable=False)
    pass_rate = Column(Float, nullable=True)
    trained_on_examples = Column(Integer, nullable=True)
    deployed_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class AdapterTrainingState(Base):
    """Scheduler bookkeeping for per-household adapter training.

    One row per household. Tracks the cutoff of the last training window,
    how many examples that window contained, and a lock flag so overlapping
    scheduler ticks don't double-enqueue.
    """
    __tablename__ = "adapter_training_state"

    household_id = Column(String(255), primary_key=True)
    last_trained_at = Column(DateTime, nullable=True)
    last_cutoff_at = Column(DateTime, nullable=True)
    last_example_count = Column(Integer, nullable=False, default=0)
    is_training = Column(Boolean, nullable=False, default=False, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class AdapterHistory(Base):
    """Append-only audit log of adapter deployments.

    Rollback works by selecting the most recent prior row for a household
    and upserting it back into active_adapter.
    """
    __tablename__ = "adapter_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    household_id = Column(String(255), nullable=False, index=True)
    adapter_hash = Column(String(128), nullable=False)
    pass_rate = Column(Float, nullable=True)
    trained_on_examples = Column(Integer, nullable=True)
    deployed_at = Column(DateTime, nullable=False)
    replaced_at = Column(DateTime, nullable=True)
    trigger = Column(String(32), nullable=False, default="scheduler")


class AdapterProposal(Base):
    """User-approval queue for trained adapters (Phase 7.1).

    Scheduler writes pending rows on eval PASS; the mobile app renders them as
    inbox items; apply/dismiss/revert API endpoints mutate status. One row per
    training run; the paired provider_name_after is the prompt-provider the
    adapter was trained against.
    """
    __tablename__ = "adapter_proposals"

    id = Column(String(36), primary_key=True)
    household_id = Column(String(255), nullable=False, index=True)
    adapter_hash = Column(String(128), nullable=False)
    provider_name_before = Column(String(255), nullable=True)
    provider_name_after = Column(String(255), nullable=True)
    pass_rate_before = Column(Float, nullable=True)
    pass_rate_after = Column(Float, nullable=True)
    latency_before_s = Column(Float, nullable=True)
    latency_after_s = Column(Float, nullable=True)
    per_command_delta_json = Column(Text, nullable=True)
    trained_on_examples = Column(Integer, nullable=True)
    # pending | applied | dismissed | expired | superseded
    status = Column(String(20), nullable=False, default="pending", index=True)
    inbox_item_id = Column(String(36), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    decided_at = Column(DateTime, nullable=True)


class BluetoothScanRequest(Base):
    """Bluetooth scan request: mobile/voice → CC → MQTT → node → CC → mobile.

    Lifecycle:
    1. Mobile (or voice command) POSTs to request a scan (status=pending, expires_at=now+2min)
    2. CC publishes MQTT to node's bluetooth-scan topic
    3. Node runs bluetoothctl scan, POSTs results back
    4. Mobile polls for results
    """
    __tablename__ = 'bluetooth_scan_requests'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    node_id = Column(String, ForeignKey('nodes.node_id', ondelete='CASCADE'), nullable=False)
    household_id = Column(String(255), nullable=False, index=True)
    status = Column(String(20), nullable=False, default="pending")  # pending, completed, failed, expired
    role = Column(String(20), nullable=False, default="source")  # source (speaker) or sink (audio input)
    results_json = Column(Text, nullable=True)  # JSON array of discovered BT devices
    device_count = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    source = Column(String(20), nullable=False, default="mobile")  # mobile or voice
    user_id = Column(Integer, nullable=True)  # speaker user ID (if voice-triggered)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    node = relationship("Node", backref=backref("bluetooth_scan_requests", passive_deletes=True), passive_deletes=True)


class BluetoothPairRequest(Base):
    """Bluetooth pair request: mobile → CC → MQTT → node → CC → mobile.

    Lifecycle:
    1. Mobile POSTs with mac_address and role (status=pending, expires_at=now+2min)
    2. CC publishes MQTT to node's bluetooth-pair topic
    3. Node runs pair+connect+audio route, POSTs result back
    4. Mobile polls for result
    """
    __tablename__ = 'bluetooth_pair_requests'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    node_id = Column(String, ForeignKey('nodes.node_id', ondelete='CASCADE'), nullable=False)
    household_id = Column(String(255), nullable=False, index=True)
    mac_address = Column(String(17), nullable=False)  # XX:XX:XX:XX:XX:XX
    role = Column(String(20), nullable=False, default="source")  # source or sink
    status = Column(String(20), nullable=False, default="pending")  # pending, completed, failed, expired
    device_name = Column(String(255), nullable=True)  # populated on success
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    node = relationship("Node", backref=backref("bluetooth_pair_requests", passive_deletes=True), passive_deletes=True)


class RequestTrace(Base):
    """Persisted request trace for the trace visualizer.

    Each row captures the full service-to-service path of a single
    voice or chat request, with per-hop timing stored as a JSON array
    of span objects in ``spans_json``.
    """
    __tablename__ = 'request_traces'

    id = Column(String(36), primary_key=True)
    conversation_id = Column(String(255), nullable=False, index=True)
    request_type = Column(String(20), nullable=False)   # warmup, voice_command, mobile_chat
    source = Column(String(20), nullable=False)          # node, mobile
    node_id = Column(String, ForeignKey('nodes.node_id', ondelete='SET NULL'), nullable=True)
    household_id = Column(String(255), nullable=True, index=True)
    user_command = Column(Text, nullable=True)
    assistant_message = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, server_default="ok")
    error_message = Column(Text, nullable=True)
    total_duration_ms = Column(Float, nullable=False)
    spans_json = Column(Text, nullable=False)            # JSON array of span objects
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)


class CallbackJob(Base):
    """Interactive-notification callback: mobile tap → CC → executor → CC → mobile.

    Two dispatch planes, distinguished by ``node_id``:

    **Node plane** (node_id set):
    1. Mobile app POSTs (user-JWT) with {command, callback, data, target_node_id}
       — created from a tap on an interactive element in a rich inbox item.
    2. CC creates the row (status=pending) and publishes
       [{"command": "callback", "details": {"request_id": id}}] to the node's
       commands topic via NodeCommandService (job.id == MQTT request_id).
    3. Node GETs /api/v0/callbacks/{id} (node X-API-Key) and reads the full
       payload. MQTT carries only the opaque id — zero-trust delivery.
    4. Node dispatches to the command's @callback-decorated method, then POSTs
       /api/v0/callbacks/{id}/result with success/error/context_data.

    **Server plane** (node_id NULL): the element belongs to a CC server tool
    (deep-research follow-ups, phone-call confirm/escalation cards). CC
    executes a handler registered in app.services.server_callback_registry
    in a background task right after creation and records the result through
    the same path. Nodes can never read or complete a server-plane job.

    Either way, the callback handler is responsible for emitting any
    follow-up inbox item (via the notifications service) — this row just
    tracks the job.
    """
    __tablename__ = 'callback_jobs'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    # NULL → server-plane job (executed in-process by CC, no node involved).
    node_id = Column(String, ForeignKey('nodes.node_id', ondelete='CASCADE'), nullable=True)
    household_id = Column(String(255), nullable=False, index=True)
    user_id = Column(Integer, nullable=True)  # JWT user who originated the tap
    command_name = Column(String(128), nullable=False)
    callback_name = Column(String(128), nullable=False)
    data_json = Column(Text, nullable=True)  # opaque per-callback payload from the inbox item
    # Stable dedup handle for proposable-action jobs (e.g. "appt:<message_id>").
    # NULL for ordinary callback jobs. The dispatcher short-circuits a second job
    # with the same (household_id, idempotency_key) that already completed, so a
    # double-tap / re-post never runs the target callback twice. (The command
    # itself must still no-op on a timeout-after-success — see ProposableAction.)
    idempotency_key = Column(String(255), nullable=True, index=True)
    # Renderer hint chosen by the command at element-creation time.
    # "new_notification" → result endpoint creates an inbox item server-side
    #                      (back-compat default; mobile fire-and-forget).
    # "stack"            → mobile pushed a screen; it polls the user-JWT'd
    #                      GET endpoint and renders inline. No inbox row.
    # "popover"          → modal sheet, same poll/render semantics as "stack".
    navigation_type = Column(String(32), nullable=False, default="new_notification")
    status = Column(String(20), nullable=False, default="pending")  # pending, completed, failed, expired
    error_message = Column(Text, nullable=True)
    result_context_data_json = Column(Text, nullable=True)  # CommandResponse.context_data from the node
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    node = relationship("Node", backref=backref("callback_jobs", passive_deletes=True), passive_deletes=True)


class Routine(Base):
    """A named bundle of commands triggered by a phrase, run-now, or a schedule.

    Source of truth for routines lives here (per household). Mobile is a thin
    CRUD client; nodes pull the household set (pull-on-nudge) into their local
    CommandDataRepository and execute on-node via the RoutineCommand engine.

    JSON-bearing columns are Text (project convention — see protocols/results_json):
    - trigger_phrases: JSON array of strings
    - steps: JSON array of {command, args:[{key,value}], label} in the MOBILE-native
      shape (args is an ARRAY of pairs). The node-pull endpoint flattens args to an
      object {k:v} at the boundary. Stored mobile-native so the editor round-trips.
    - schedule: JSON object or null. null = voice/run-now only. Shape:
      {type:"cron"|"interval", cron?:str, interval_seconds?:int, timezone:str,
       target_node_id:str, enabled:bool, last_fired_at?:str(iso)}.
    """
    __tablename__ = 'routines'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    household_id = Column(String(255), nullable=False, index=True)
    # Node-facing key (the CommandDataRepository data_key). Stable across renames.
    slug = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    trigger_phrases = Column(Text, nullable=False, default="[]")
    steps = Column(Text, nullable=False, default="[]")
    response_instruction = Column(Text, nullable=False, default="")
    response_length = Column(String(16), nullable=False, default="short")  # short|medium|long
    schedule = Column(Text, nullable=True)  # JSON object or NULL
    enabled = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('household_id', 'slug', name='uq_routine_household_slug'),
    )


class RoutineExecution(Base):
    """Audit trail of routine runs (run-now + scheduled). Voice runs may also be
    recorded if the node reports them. TTL-cleaned by a background worker."""
    __tablename__ = 'routine_executions'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    routine_id = Column(String(36), ForeignKey('routines.id', ondelete='CASCADE'), nullable=True, index=True)
    household_id = Column(String(255), nullable=False, index=True)
    node_id = Column(String, nullable=True)
    trigger = Column(String(16), nullable=False)  # voice|run_now|scheduled
    status = Column(String(16), nullable=False)    # success|partial|failed|timeout
    passed = Column(Integer, nullable=False, default=0)
    failed = Column(Integer, nullable=False, default=0)
    error = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)


class AttentionEvent(Base):
    """One proactive notification request, as received (pre-gating).

    Recorded for every request that reaches an interposed endpoint (or
    /api/v0/attention/events) while attention.enabled is on for the household.
    household_id always comes from the validated node row / app auth — never
    the payload. See prds/attention-broker.md.
    """
    __tablename__ = 'attention_events'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    household_id = Column(String(255), nullable=False, index=True)
    # Phase 1: source == the request's category (category-grade identity).
    # An SDK-declared, registry-validated source name replaces this later.
    source = Column(String(100), nullable=False)
    category = Column(String(50), nullable=False)
    title = Column(String(500), nullable=False)
    summary = Column(Text, nullable=False, default="")
    dedupe_key = Column(String(255), nullable=True, index=True)
    target_user_id = Column(Integer, nullable=True)  # NULL = household-wide
    origin_node_id = Column(String(255), nullable=True)
    payload_json = Column(Text, nullable=True)  # original request, for replay
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class AttentionDelivery(Base):
    """The gated outcome of one AttentionEvent: which rung, and why.

    gate_trail_json is the ordered record of every gate decision — the
    substrate for "why did you tell me that?" and the withheld-items journal.
    request_id is reserved for the future verified in-room rung and is
    DB-backed deliberately (NodeCommandService's pending store is in-memory,
    5-minute TTL, per-process — unusable for durable invites).
    """
    __tablename__ = 'attention_deliveries'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    event_id = Column(String(36), ForeignKey('attention_events.id', ondelete='CASCADE'), nullable=False, index=True)
    household_id = Column(String(255), nullable=False, index=True)
    rung = Column(String(20), nullable=False)  # journal|inbox|push (later: led_invite|speak)
    gate_trail_json = Column(Text, nullable=False, default="[]")
    withheld_by = Column(String(50), nullable=True)  # gate name when rung == journal
    inbox_item_id = Column(String(36), nullable=True)
    request_id = Column(String(36), nullable=True, index=True)  # future in-room rung
    outcome = Column(String(30), nullable=True)  # delivered|redeemed|expired|failed|duplicate
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)


class AttentionSourceTier(Base):
    """Earned speaking rights per (household, source, category). Phase 2.

    Sources start at T1 (inbox-only probation) and move on decayed feedback
    scores. state_reason is human-readable, mirroring the node agent
    auto-disable culture: demotion always says why.
    """
    __tablename__ = 'attention_source_tiers'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    household_id = Column(String(255), nullable=False, index=True)
    source = Column(String(100), nullable=False)
    category = Column(String(50), nullable=False)
    tier = Column(Integer, nullable=False, default=1)  # 0=journal-only .. 3=push; 4/5 reserved
    score = Column(Float, nullable=False, default=0.0)
    state_reason = Column(String(255), nullable=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('household_id', 'source', 'category', name='uq_attention_tier_scope'),
    )


class AttentionConsent(Base):
    """User-granted ceiling per (household, source, category). Phase 3.

    max_rung is the highest rung the source may occupy. Writes are
    household-admin gated (mobile_household_settings pattern), never the
    superuser settings router. Speech is granted here only — never a default.
    """
    __tablename__ = 'attention_consents'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    household_id = Column(String(255), nullable=False, index=True)
    source = Column(String(100), nullable=False)
    category = Column(String(50), nullable=False)
    max_rung = Column(String(20), nullable=False, default="push")  # never|inbox|push|led_invite|speak
    granted_by_user_id = Column(Integer, nullable=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('household_id', 'source', 'category', name='uq_attention_consent_scope'),
    )


class AttentionFeedback(Base):
    """One user reaction to one delivery (button tap or voice verb). Phase 2."""
    __tablename__ = 'attention_feedback'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    delivery_id = Column(String(36), ForeignKey('attention_deliveries.id', ondelete='CASCADE'), nullable=False, index=True)
    user_id = Column(Integer, nullable=True)
    verb = Column(String(20), nullable=False)  # useful|not_useful|mute|why
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class PhoneCallSession(Base):
    """AI phone-call session — NodeTask-style state machine (phone-calls PRD).

    States: draft → confirmed → dialing → in_call → wrapup →
    done | failed | declined | expired.

    Written BEFORE dialing (the anti-deep-research rule: no session ever
    vanishes). CC owns this row exclusively; the gateway reports through
    POST /internal/phone/sessions/{id}/events and never touches the DB.

    Audit trail (security requirement 7): resolved_number is what the
    resolver produced, dialed_number is what was actually confirmed after
    the user-editable card — number_edited says they differ; confirmed_by
    is the JWT user whose tap authorized the dial (may differ from the
    initiating speaker).
    """
    __tablename__ = 'phone_call_sessions'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    household_id = Column(String(255), nullable=False, index=True)
    user_id = Column(Integer, nullable=True)          # initiating speaker/user
    confirmed_by = Column(Integer, nullable=True)     # JWT user who tapped confirm
    contact_id = Column(String(36), ForeignKey('phone_contacts.id', ondelete='SET NULL'), nullable=True)
    contact_name = Column(String(255), nullable=True)
    # Address discovered during resolution (web search); carried to the
    # phonebook row auto-save writes when the call succeeds.
    contact_address = Column(Text, nullable=True)
    # When an errand step placed this call, link back so the call's terminal outcome
    # (done/failed/…) can resume the waiting errand (durable wait-and-decide).
    errand_id = Column(String(40), nullable=True, index=True)
    errand_step = Column(Integer, nullable=True)      # index of the errand step that placed it
    goal = Column(Text, nullable=False)               # what the call should achieve
    details = Column(Text, nullable=True)             # the confirmed brief (guardrail boundary)
    constraints = Column(Text, nullable=True)         # constraint envelope (P2)
    resolved_number = Column(String(32), nullable=True)
    dialed_number = Column(String(32), nullable=True)
    number_edited = Column(Boolean, default=False, nullable=False)
    line_type = Column(String(16), nullable=True)     # mobile|landline|voip|unknown
    state = Column(String(16), nullable=False, default="draft", index=True)
    error_message = Column(Text, nullable=True)
    transcript_json = Column(Text, nullable=True)     # JSON list of turn dicts + timings
    outcome_json = Column(Text, nullable=True)        # structured facts from wrapup
    audio_object_key = Column(String(512), nullable=True)  # MinIO key (local mix)
    worker_url = Column(String(512), nullable=True)   # claiming gateway worker (session affinity)
    heartbeat_at = Column(DateTime, nullable=True)
    twilio_call_sid = Column(String(64), nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    confirmed_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)      # draft TTL (plan_ttl_minutes)
    ended_at = Column(DateTime, nullable=True)


class PhoneContact(Base):
    """Household phonebook entry (phone-calls PRD data model).

    Household-scoped: a business's number is a household fact. Per-user
    overlays (usual order, preferences) live in overlay_json keyed by
    user_id. do_not_call is enforced at resolve time — a DNC'd contact is
    refused before a plan is ever drafted.
    """
    __tablename__ = 'phone_contacts'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    household_id = Column(String(255), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    normalized_name = Column(String(255), nullable=False)
    number = Column(String(32), nullable=False)       # E.164
    address = Column(Text, nullable=True)
    source = Column(String(16), nullable=False, default="manual")  # manual|call|web
    line_type = Column(String(16), nullable=True)
    do_not_call = Column(Boolean, default=False, nullable=False)
    notes = Column(Text, nullable=True)
    overlay_json = Column(Text, nullable=True)        # per-user overlays keyed by user_id
    verified_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint('household_id', 'normalized_name', name='uq_phone_contact_household_name'),
    )


class ErrandPlan(Base):
    """A drafted, reviewable errand plan (Errand Runner POC — errand-runner PRD §2-§3).

    "LLM proposes, card disposes": the background planner drafts routine-shaped
    steps from a natural-language goal, we persist them here as a DRAFT, and the
    plan card is the trust checkpoint — nothing runs until the user taps Run. CC
    owns this row exclusively; the node holds no errand state (PRD §2.4). The
    lifecycle mirrors the phone-call state machine (draft → confirmed → running →
    done | failed, or cancelled/expired), with `transition()` semantics owned by
    the callback layer rather than the DB.

    JSON-bearing columns are Text (project convention — see Routine.steps):
    - steps: JSON array of node-native {command, args:{k:v}, label} dicts. This IS
      the plan — the errand executor (errand_executor.execute_errand) dispatches
      each step to its plane at Run time (node command → the node over MQTT; server
      tool → in-process). Produced by errand_planner.plan_errand(); the planner
      validates every command against its menu, so hallucinated steps never land here.

    `node_id` is the target node for the plan's NODE-command steps (server-tool
    steps run in CC). `revision` backs the plan card's stale-approval guard: an
    approve callback carrying an old revision is rejected, exactly like the phone
    card refuses a non-draft session.
    """
    __tablename__ = 'errand_plans'

    id = Column(String(40), primary_key=True, default=lambda: f"pl_{uuid4().hex}")
    household_id = Column(String(255), nullable=False, index=True)
    user_id = Column(Integer, nullable=True)          # initiating speaker; completion-notify target
    node_id = Column(String, nullable=True)           # target node for the plan's node-command steps
    goal = Column(Text, nullable=False)               # the user's natural-language errand goal
    summary = Column(Text, nullable=True)             # planner's one-line plain-English plan
    steps = Column(Text, nullable=False, default="[]")  # JSON array of node-native steps — the plan itself
    routine_slug = Column(String(255), nullable=True)   # vestigial: no transient routine since per-step dispatch (unused)
    inbox_item_id = Column(String(64), nullable=True)   # the plan card's inbox item — updated in place on revise
    state = Column(String(16), nullable=False, default="draft", index=True)  # draft|running|waiting|done|partial|failed|timeout|cancelled|expired
    revision = Column(Integer, nullable=False, default=1)  # bumps on edit; stale-approval guard
    # Durable wait-and-decide execution: when a step places a phone call the errand
    # SUSPENDS (state="waiting") until the call finishes, then resumes from `cursor`.
    cursor = Column(Integer, nullable=False, default=0)          # next step index to run on resume
    results_json = Column(Text, nullable=False, default="[]")    # accumulated per-step outcomes across suspends
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    confirmed_at = Column(DateTime, nullable=True)    # when the user tapped Run
    expires_at = Column(DateTime, nullable=True)      # draft TTL
    # The run/instance split (errand-runner PRD §2.5): once the user taps Run, the
    # DRAFT (this row) spawns a Workflow RUN and links to it here; the run's
    # execution state (running/waiting/done/…) lives on the Workflow, not here. This
    # is what lets ONE definition drive MANY runs (scheduling fast-follow). The
    # `cursor`/`results_json`/running-side `state` values above are vestigial after
    # the split (kept, not dropped) — the Workflow owns them now.
    workflow_id = Column(String(40), nullable=True, index=True)  # the run created on Run


class Workflow(Base):
    """A durable workflow RUN / INSTANCE — the engine's unit of execution (errand-
    runner PRD §2.5-2.6). Extracted from ``ErrandPlan`` in the run/instance split:
    an ``ErrandPlan`` is the DRAFT/DEFINITION (goal, steps, revision, plan card);
    a ``Workflow`` is ONE run of a plan. Errands are the first consumer (``kind``),
    but the table is consumer-agnostic so scheduling (one definition → many runs)
    and future workflow kinds drop in without re-modeling.

    The engine (``workflow_engine``) owns this row's lifecycle via the ``WorkflowStore``
    seam: it runs the plan step-by-step, SUSPENDS (``state='waiting'``) on a deferred
    step (a phone call) until an external signal resumes it, and lands terminal. The
    recovery/timeout safety sweep operates on THIS table, so every consumer gets
    restart-durability for free.

    JSON-bearing columns are Text (project convention):
    - steps: JSON array of node-native {command, args:{k:v}, label} — the plan running.
    - results_json: accumulated per-step outcomes across suspends (for the summary).
    """
    __tablename__ = 'workflows'

    id = Column(String(40), primary_key=True, default=lambda: f"wf_{uuid4().hex}")
    kind = Column(String(40), nullable=False, index=True)  # "errand" today; selects the consumer
    household_id = Column(String(255), nullable=False, index=True)
    user_id = Column(Integer, nullable=True)          # initiating speaker; completion-notify target
    node_id = Column(String, nullable=True)           # target node for the plan's node-command steps
    goal = Column(Text, nullable=False, default="")   # the objective, for the completion summary
    title = Column(Text, nullable=True)               # display label (the plan summary)
    steps = Column(Text, nullable=False, default="[]")  # JSON array of node-native steps — the plan
    cursor = Column(Integer, nullable=False, default=0)          # next step index to run on resume
    results_json = Column(Text, nullable=False, default="[]")    # accumulated per-step outcomes
    # Bumps on a mid-run replan (a request_replan step splices new steps in).
    # The change/delta card carries the revision it was posted for, so a tap on
    # a stale card (one the run already moved past) is rejected — same
    # stale-guard the draft ErrandPlan uses.
    revision = Column(Integer, nullable=False, default=1)
    # The run's live card (plan/delta/status) in the notifications inbox, so an
    # update posts IN PLACE instead of stacking a second card.
    inbox_item_id = Column(String, nullable=True)
    # running | waiting | done | partial | failed | timeout | cancelled
    state = Column(String(16), nullable=False, default="running", index=True)
    # When state == 'waiting', which wakeup source will resume it, and (for a timer)
    # when. ``phone_call`` waits resume on the call's outcome; ``timer`` waits
    # resume when ``wake_at`` passes (the wait_for step); ``approval`` waits on a
    # human tapping the mid-run replan card (the request_replan step). Persisted
    # so a wait survives a restart — the point of a durable engine.
    waiting_on = Column(String(20), nullable=True)  # phone_call | timer | approval
    wake_at = Column(DateTime, nullable=True, index=True)  # timer wait: UTC instant to resume
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class Schedule(Base):
    """A durable TRIGGER that fires the plan→card loop on a clock. A schedule is NOT a
    plan or a run — at ``next_fire_at`` a sweep re-plans ``intent`` against fresh context
    and posts a plan card the user must approve (design decision: re-plan + RE-CONFIRM
    each run, so a schedule never acts autonomously). One-shot for now (``recurrence``
    null); a recurring schedule re-arms ``next_fire_at`` after each fire.

    ``timezone`` is the user's IANA zone — kept so recurrence math (e.g. "every Monday
    8am") stays correct across DST; ``next_fire_at`` itself is a naive-UTC instant to
    match ``workflows.wake_at`` and the due sweep. A later ``silent`` mode (per-schedule
    opt-in, gated by blast-radius) would skip the card for low-risk recurrences.
    """

    __tablename__ = 'schedules'

    id = Column(String(40), primary_key=True, default=lambda: f"sch_{uuid4().hex}")
    household_id = Column(String(255), nullable=False, index=True)
    user_id = Column(Integer, nullable=True)   # who scheduled it / plan-card + notify target
    node_id = Column(String, nullable=True)    # target node for the planned errand's node steps
    intent = Column(Text, nullable=False)      # the natural-language goal to re-plan at fire time
    title = Column(Text, nullable=True)        # display label
    timezone = Column(String(64), nullable=False, default="UTC")  # IANA zone for recurrence math
    next_fire_at = Column(DateTime, nullable=False, index=True)    # naive UTC — the due sweep key
    recurrence = Column(Text, nullable=True)   # JSON recurrence spec; NULL = one-shot (Slice 1)
    # active | done | cancelled | paused
    state = Column(String(16), nullable=False, default="active", index=True)
    last_fired_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class ProposalSuppression(Base):
    """A user's "never suggest this again" blocklist entry for proposable actions.

    Written by the ``jarvis.proposable_action.suppress`` handler when a user taps
    "Never suggest this" on a proposable-action card; consulted by the detector
    agent each run — it deterministically skips a candidate whose ``source_key``
    matches, and injects ``descriptor`` into its extraction prompt as a negative
    example so the LLM also skips look-alikes. Scoped per (household, user,
    command).
    """

    __tablename__ = "proposal_suppressions"

    id = Column(String(40), primary_key=True, default=lambda: f"sup_{uuid4().hex}")
    household_id = Column(String(255), nullable=False, index=True)
    user_id = Column(Integer, nullable=True, index=True)   # whose blocklist this is
    command = Column(String(128), nullable=False)          # target command, e.g. "add_event"
    source_key = Column(String(512), nullable=True)        # deterministic hard key (e.g. sender)
    descriptor = Column(Text, nullable=True)               # semantic example for prompt injection
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
