import time
import hashlib
from typing import Dict
import streamlit as st
from functools import wraps

class SecurityManager:
    """Enhanced security for MarketShield with rate limiting and API key protection."""

    def __init__(self):
        self.request_history: Dict[str, list[float]] = {}

        # Define all action buckets used by decorators across the app.
        # Adding 'file_analysis' fixes the KeyError you observed.
        self.rate_limits: Dict[str, Dict[str, int]] = {
            'analysis':      {'max_requests': 10, 'window': 300},  # 10 per 5 minutes
            'url_fetch':     {'max_requests': 5,  'window': 300},  # 5 per 5 minutes
            'file_analysis': {'max_requests': 5,  'window': 300},  # uploads
            # Optional future buckets:
            # 'youtube_transcript': {'max_requests': 5, 'window': 300},
        }

        # Default limit if a decorator uses a new action not pre-registered.
        self.default_limit = {'max_requests': 10, 'window': 300}

    def check_rate_limit(self, user_id: str, action: str) -> bool:
        """Return True if the call is allowed under the current rate limits; False if blocked."""
        current_time = time.time()
        key = f"{user_id}_{action}"

        # Use .get() to avoid KeyError if an unknown action name is passed.
        limits = self.rate_limits.get(action, self.default_limit)
        window = limits['window']
        max_requests = limits['max_requests']

        # Initialize per-key history
        if key not in self.request_history:
            self.request_history[key] = []

        # Drop timestamps outside the window
        self.request_history[key] = [t for t in self.request_history[key] if current_time - t < window]

        # Enforce limit
        if len(self.request_history[key]) >= max_requests:
            return False

        # Record this call
        self.request_history[key].append(current_time)
        return True

    def get_user_id(self) -> str:
        """Generate a per-session user ID for rate limiting and audit."""
        if 'user_id' not in st.session_state:
            session_info = f"{st.session_state.get('session_id', 'anonymous')}_{time.time()}"
            st.session_state.user_id = hashlib.md5(session_info.encode()).hexdigest()
        return st.session_state.user_id

    def encrypt_api_key(self, api_key: str) -> str:
        """Lightweight XOR obfuscation for API keys (demo only; use KMS/secret manager in prod)."""
        if not api_key:
            return ""
        key = "MARKETSHIELD_SECRET_2025"
        return ''.join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(api_key))

    def decrypt_api_key(self, encrypted_key: str) -> str:
        """Reverse XOR obfuscation."""
        if not encrypted_key:
            return ""
        key = "MARKETSHIELD_SECRET_2025"
        return ''.join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(encrypted_key))

def rate_limit(action: str):
    """Decorator to apply named rate limits to any function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            security_manager = st.session_state.get('security_manager')
            if not security_manager:
                security_manager = SecurityManager()
                st.session_state.security_manager = security_manager

            user_id = security_manager.get_user_id()
            if not security_manager.check_rate_limit(user_id, action):
                st.error(f"⚠️ Rate limit exceeded for {action}. Please wait and try again.")
                return None
            return func(*args, **kwargs)
        return wrapper
    return decorator
