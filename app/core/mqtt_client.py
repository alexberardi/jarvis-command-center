"""
MQTT client for publishing messages to nodes.

Uses paho-mqtt to connect to Mosquitto broker.
Broker URL is discovered via service config or environment variable.
"""
import logging
import os
import uuid
from typing import Optional

import paho.mqtt.client as mqtt

logger = logging.getLogger("uvicorn")


def get_mqtt_broker_url() -> Optional[str]:
    """
    Get MQTT broker URL from service discovery or environment.

    Returns URL in format: mqtt://host:port or None if not configured.

    Resolution order:
    1. jarvis-config-service (if initialized)
    2. JARVIS_MQTT_BROKER_URL env var
    3. Default: mqtt://localhost:1883
    """
    try:
        from . import service_config
        if service_config.is_initialized():
            return service_config.get_mqtt_broker_url()
    except Exception as e:
        logger.debug(f"Service discovery not available: {e}")

    # Fallback to environment variable or default
    return os.getenv("JARVIS_MQTT_BROKER_URL", "mqtt://localhost:1883")


def parse_mqtt_url(url: str) -> tuple[str, int, str]:
    """
    Parse MQTT URL into host, port, and transport.

    Supports formats:
    - mqtt://host:port   → TCP transport
    - mqtts://host:port  → TCP + TLS
    - ws://host:port     → WebSocket transport
    - wss://host:port    → WebSocket + TLS
    - host:port          → TCP transport (default)
    - host               → TCP transport, port 1883

    Returns:
        (host, port, transport) where transport is "tcp" or "websockets"
    """
    scheme = "mqtt"
    for prefix in ("wss://", "ws://", "mqtts://", "mqtt://"):
        if url.startswith(prefix):
            scheme = prefix.rstrip(":/")
            url = url[len(prefix):]
            break

    if ":" in url:
        host, port_str = url.rsplit(":", 1)
        port = int(port_str)
    else:
        host = url
        port = 443 if scheme in ("wss", "mqtts") else 1883

    transport = "websockets" if scheme in ("ws", "wss") else "tcp"
    return host, port, transport


class MQTTClient:
    """
    MQTT client wrapper for publishing messages to nodes.

    Thread-safe, maintains persistent connection with auto-reconnect.
    """

    def __init__(self, broker_url: str):
        self.broker_url = broker_url
        self.host, self.port, self.transport = parse_mqtt_url(broker_url)
        self._use_tls = broker_url.startswith("wss://") or broker_url.startswith("mqtts://")
        self.client: Optional[mqtt.Client] = None
        self._connected = False

    def connect(self) -> None:
        """Establish connection to MQTT broker."""
        if self.client is not None:
            return

        self.client = mqtt.Client(
            client_id=f"jarvis-cc-{os.getpid()}-{uuid.uuid4().hex[:8]}",
            protocol=mqtt.MQTTv311,
            transport=self.transport,
        )

        if self._use_tls:
            import ssl
            self.client.tls_set(cert_reqs=ssl.CERT_REQUIRED)

        # Set up callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        try:
            self.client.connect(self.host, self.port, keepalive=60)
            self.client.loop_start()  # Start background thread
            logger.info(f"✅ MQTT client connecting to {self.host}:{self.port} ({self.transport})")
        except Exception as e:
            logger.error(f"❌ Failed to connect to MQTT broker: {e}")
            self.client = None
            raise

    def _on_connect(self, client, userdata, flags, rc) -> None:
        """Callback when connected to broker."""
        if rc == 0:
            self._connected = True
            logger.info("✅ MQTT client connected")
        else:
            logger.error(f"❌ MQTT connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc) -> None:
        """Callback when disconnected from broker."""
        self._connected = False
        if rc != 0:
            logger.warning(f"MQTT client disconnected unexpectedly (rc={rc})")

    def _on_message(self, client, userdata, msg) -> None:
        """Default message handler (per-topic callbacks take priority)."""
        logger.debug(f"MQTT message on {msg.topic} (no specific handler)")

    def publish(self, topic: str, payload: str, qos: int = 1) -> None:
        """
        Publish a message to a topic.

        Args:
            topic: MQTT topic (e.g., "jarvis/nodes/{node_id}/settings/request")
            payload: JSON string payload
            qos: Quality of Service level (0, 1, or 2)

        Raises:
            RuntimeError: If not connected to broker
        """
        if self.client is None:
            raise RuntimeError("MQTT client not initialized")

        result = self.client.publish(topic, payload, qos=qos)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(f"MQTT publish failed with code {result.rc}")

        logger.debug(f"📤 MQTT published to {topic}")

    def subscribe_and_wait(
        self, topic: str, timeout: float = 10.0,
    ) -> Optional[str]:
        """Subscribe to a topic, wait for one message, then unsubscribe.

        Args:
            topic: MQTT topic to subscribe to.
            timeout: Max seconds to wait for a message.

        Returns:
            The message payload as a string, or None if timed out.
        """
        import threading

        if self.client is None:
            return None

        result_holder: list[str] = []
        event = threading.Event()

        def on_message(_client: mqtt.Client, _userdata: object, msg: mqtt.MQTTMessage) -> None:
            result_holder.append(msg.payload.decode("utf-8", errors="replace"))
            event.set()

        self.client.subscribe(topic, qos=1)
        self.client.message_callback_add(topic, on_message)

        try:
            event.wait(timeout=timeout)
        finally:
            self.client.message_callback_remove(topic)
            self.client.unsubscribe(topic)

        return result_holder[0] if result_holder else None

    def request_response(
        self,
        request_topic: str,
        response_topic: str,
        payload: str,
        timeout: float = 10.0,
    ) -> Optional[str]:
        """Synchronous request/response round-trip with a node.

        Subscribes to `response_topic` BEFORE publishing the request to
        avoid the race where the node responds faster than we can
        subscribe. Used by the mobile command-data browser, where mobile
        is sitting and waiting for an interactive CRUD response — the
        file-callback pattern (see `bluetooth.py`, `node_commands.py`)
        adds 100ms polling latency that hurts UX here.

        Args:
            request_topic: Topic to publish the request to (node listens).
            response_topic: Topic to subscribe to for the response. Should
                be unique per-request (e.g. include a correlation_id) so
                concurrent in-flight requests don't collide.
            payload: JSON string to publish.
            timeout: Max seconds to wait for the response.

        Returns:
            The response payload as a string, or None if timed out.
        """
        import threading

        if self.client is None:
            return None

        result_holder: list[str] = []
        event = threading.Event()

        def on_message(_client: mqtt.Client, _userdata: object, msg: mqtt.MQTTMessage) -> None:
            result_holder.append(msg.payload.decode("utf-8", errors="replace"))
            event.set()

        self.client.subscribe(response_topic, qos=1)
        self.client.message_callback_add(response_topic, on_message)

        try:
            self.client.publish(request_topic, payload, qos=1)
            event.wait(timeout=timeout)
        finally:
            self.client.message_callback_remove(response_topic)
            self.client.unsubscribe(response_topic)

        return result_holder[0] if result_holder else None

    def disconnect(self) -> None:
        """Disconnect from broker."""
        if self.client is not None:
            self.client.loop_stop()
            self.client.disconnect()
            self.client = None
            self._connected = False
            logger.info("MQTT client disconnected")

    @property
    def is_connected(self) -> bool:
        """Check if connected to broker."""
        return self._connected
