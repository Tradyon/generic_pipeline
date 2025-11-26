"""
Utility functions for attribute classifier.
"""

import json
import logging

logger = logging.getLogger(__name__)

class TokenTracker:
    """Track token usage against a budget."""
    
    def __init__(self, budget: int):
        self.budget = int(budget)
        self.used = 0
    
    def add(self, tokens: int):
        self.used += int(tokens)
        return self.used <= self.budget
    
    def remaining(self) -> int:
        return max(0, self.budget - self.used)
    
    def usage(self) -> int:
        return self.used

def parse_json_response(text: str) -> dict:
    """Parse JSON from response text."""
    if not text:
        return {}
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start >= 0 and end >= 0:
            return json.loads(text[start:end+1])
    except Exception:
        return {}
    return {}

def sanitize_filename(name: str) -> str:
    """Sanitize string for use in filenames."""
    safe = ''.join(ch if ch.isalnum() or ch in (' ', '_', '-') else '_' for ch in name)
    safe = '_'.join(safe.strip().split())
    return safe[:120] or 'product'
