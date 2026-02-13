"""
Debug setup module for handling debugpy configuration.
This module handles the optional debugpy import and setup to avoid linter errors.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger("uvicorn")

def setup_debugger(host: str = "0.0.0.0", port: Optional[int] = None) -> Optional[bool]:
    """
    Set up debugpy debugger if DEBUG environment variable is set.
    
    Args:
        host: Host to bind debugger to
        port: Port to bind debugger to
        
    Returns:
        True if debugger was set up successfully, False if debugpy not available,
        None if DEBUG not enabled
    """
    if not os.getenv("DEBUG", "").lower() in ("true", "1", "yes"):
        return None

    if port is None:
        debug_port = os.getenv("DEBUG_PORT", "5678")
        try:
            port = int(debug_port)
        except ValueError:
            logger.warning("Invalid DEBUG_PORT '%s', falling back to 5678", debug_port)
            port = 5678
    
    try:
        import debugpy
        debugpy.listen((host, port))
        logger.info("Debugger listening on %s:%s", host, port)
        return True
    except ImportError:
        logger.warning("debugpy not available - debugging disabled")
        return False
    except Exception as e:
        logger.warning("Failed to start debugger: %s", e)
        return False 