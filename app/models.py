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
    status = Column(String(20), nullable=False, default="pending")  # pending, completed, failed, expired
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
