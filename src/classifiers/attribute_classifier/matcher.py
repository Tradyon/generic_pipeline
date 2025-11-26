"""
Deterministic matcher for attribute classification.
"""

import re
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

# Negation detection constants
NEGATION_WORDS = [
    'not', 'no', 'without', 'w/o', 'minus', 'excluding', 'free of', 'free-from', 'non'
]
NEGATION_PREFIXES = ['un', 'non']


class DeterministicMatcher:
    """
    Robust deterministic multi-attribute matcher with negation awareness.
    
    Features:
    - Whole-token & multiword value matching (case-insensitive)
    - Longest-value precedence
    - Negation detection (rejects matches in negated phrases)
    - Boundary-aware matching
    """
    
    def __init__(self, attr_values: Dict[str, List[str]], config: Dict[str, Any]):
        self.attr_values = attr_values
        self.config = config
        self.min_len = config.get('deterministic_min_token_chars', 3)
        self.patterns: Dict[str, List[Tuple[str, re.Pattern, int]]] = {}
        self._compile()
        self.stats = {
            'attempted': 0,
            'matched': 0,
            'negation_blocked': 0,
            'attr_attempted': 0,
            'attr_matched': 0,
        }
    
    def _compile(self):
        """Compile regex patterns for all attribute values."""
        for attr, values in self.attr_values.items():
            compiled: List[Tuple[str, re.Pattern, int]] = []
            seen_norm = set()
            for v in values:
                if not isinstance(v, str):
                    continue
                vv = v.strip()
                if len(vv) < self.min_len:
                    continue
                norm = ' '.join(vv.lower().split())
                if norm in seen_norm:
                    continue
                seen_norm.add(norm)
                
                # Handle density-like patterns (e.g., "500 G/L")
                dens_like = False
                if re.search(r"\b\d{3,4}\s*g/?l\b", norm.replace(' ', '')) or 'g/l' in norm or 'gr/l' in norm or 'grams/liter' in norm:
                    dens_like = True
                
                if dens_like:
                    number_part = re.findall(r"\d+", norm)
                    number_regex = r"(?:" + r"|".join(sorted({re.escape(n) for n in number_part}, key=len, reverse=True)) + r")" if number_part else r"\d+"
                    dens_pattern = number_regex + r"\s*(?:g|gr|grams)\s*/?\s*(?:l|liter)"
                    pattern = re.compile(rf"(?<![A-Za-z0-9]){dens_pattern}(?![A-Za-z0-9])", re.IGNORECASE)
                    compiled.append((v, pattern, len(v)))
                else:
                    escaped = re.escape(norm)
                    pattern = re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)
                    compiled.append((v, pattern, len(v)))
            
            compiled.sort(key=lambda x: (-x[2], x[0]))
            self.patterns[attr] = compiled
    
    def _has_negation(self, text: str, start: int, end: int, token_lower: str) -> bool:
        """Check if match is in a negated context."""
        window_chars = self.config.get('deterministic_negation_window_chars', 28)
        prefix_slice = text[max(0, start-4):start].lower()
        
        for pref in NEGATION_PREFIXES:
            if prefix_slice.endswith(pref) and token_lower.startswith(prefix_slice[-len(pref):]):
                return True
        
        look_back = text[max(0, start-window_chars):start].lower()
        look_back = re.sub(r'[^a-z0-9]+', ' ', look_back)
        for w in NEGATION_WORDS:
            if look_back.endswith(' ' + w) or look_back == w or look_back.endswith(w.replace(' ', '')):
                return True
        
        return False
    
    def classify_goods(self, goods_list: List[str]) -> List[Dict[str, Any]]:
        """Classify a list of goods descriptions."""
        results: List[Dict[str, Any]] = []
        for g in goods_list:
            self.stats['attempted'] += 1
            lower_txt = ' ' + ' '.join(g.lower().split()) + ' '
            attr_assign: Dict[str, str] = {}
            
            for attr, pats in self.patterns.items():
                self.stats['attr_attempted'] += 1
                chosen = None
                chosen_pos = 10**9
                
                for canonical, rgx, _length in pats:
                    m = rgx.search(lower_txt)
                    if not m:
                        continue
                    
                    if self.config.get('enable_negation_guard') and self._has_negation(lower_txt, m.start(), m.end(), canonical.lower()):
                        self.stats['negation_blocked'] += 1
                        continue
                    
                    pos = m.start()
                    if chosen is None or pos < chosen_pos:
                        chosen = canonical
                        chosen_pos = pos
                    break
                
                if chosen:
                    attr_assign[attr] = chosen
                    self.stats['attr_matched'] += 1
            
            if attr_assign:
                self.stats['matched'] += 1
            results.append({'goods_shipped': g, 'attributes': attr_assign})
        
        return results
    
    def coverage(self) -> Dict[str, Any]:
        """Return coverage statistics."""
        total_attrs = self.stats['attr_attempted']
        return {
            'rows_with_any_match': self.stats['matched'],
            'rows_total': self.stats['attempted'],
            'attribute_matches': self.stats['attr_matched'],
            'attribute_attempted': total_attrs,
            'attribute_match_rate': round(self.stats['attr_matched']/max(1, total_attrs), 6),
            'negation_blocked': self.stats['negation_blocked']
        }
