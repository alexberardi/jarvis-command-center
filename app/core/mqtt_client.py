"""
MQTT client for publishing messages to nodes.

Uses paho-mqtt to connect to Mosquitto broker.
Broker URL is discovered via service config or environment variable.
"""
import logging
import os
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


def parse_mqtt_url(url: str) -> tuple[str, int]:
    """
    Parse MQTT URL into host and port.

    Supports formats:
    - mqtt://host:port
    - host:port
    - host (defaults to port 1883)
    """
    if url.startswith("mqtt://"):
        url = url[7:]  # Remove mqtt:// prefix

    if ":" in url:
        host, port_str = url.rsplit(":", 1)
        return host, int(port_str)
    else:
        return url, 1883  # Default MQTT port


class MQTTClient:
    """
    MQTT client wrapper for publishing messages to nodes.

    Thread-safe, maintains persistent connection with auto-reconnect.
    """

    def __init__(self, broker_url: str):
        self.broker_url = broker_url
        self.host, self.port = parse_mqtt_url(broker_url)
        self.client: Optional[mqtt.Client] = None
        self._connected = False

    def connect(self) -> None:
        """Establish connection to MQTT broker."""
        if self.client is not None:
            return

        self.client = mqtt.Client(
            client_id=f"jarvis-command-center-{os.getpid()}",
            protocol=mqtt.MQTTv311,
        )

        # Set up callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

        try:
            self.client.connect(self.host, self.port, keepalive=60)
            self.client.loop_start()  # Start background thread
            logger.info(f"✅ MQTT client connecting to {self.host}:{self.port}")
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
            logger.warning(f"⚠️  MQTT client disconnected unexpectedly (rc={rc})")

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

    def disconnect(self) -> None:
        """Disconnect from broker."""
        if self.client is not None:
            self.client.loop_stop()
            self.client.disconnect()
            self.client = None
            self._connected = False
            logger.info("🔌 MQTT client disconnected")

    @property
    def is_connected(self) -> bool:
        """Check if connected to broker."""
        return self._connected
