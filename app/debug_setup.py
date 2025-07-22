"""
Debug setup module for handling debugpy configuration.
This module handles the optional debugpy import and setup to avoid linter errors.
"""

import os
from typing import Optional

def setup_debugger(host: str = "0.0.0.0", port: int = 5678) -> Optional[bool]:
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
    
    try:
        import debugpy
        debugpy.listen((host, port))
        print(f"🔧 Debugger listening on {host}:{port}")
        return True
    except ImportError:
        print("⚠️  debugpy not available - debugging disabled")
        return False
    except Exception as e:
        print(f"⚠️  Failed to start debugger: {e}")
        return False 