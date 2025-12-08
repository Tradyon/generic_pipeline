"""
LLM logic for attribute classification.
"""

import time
import logging
from typing import Dict, List, Any, Tuple, Optional

from .utils import TokenTracker, parse_json_response
from .matcher import DeterministicMatcher

logger = logging.getLogger(__name__)

def build_multiitem_response_schema(attr_values: Dict[str, List[str]], allow_out_of_schema: bool = True):
    """Build JSON schema for multi-item batch classification."""
    attr_props = {}
    for attr, vals in attr_values.items():
        if allow_out_of_schema:
            attr_props[attr] = {'type': 'string'}
        else:
            enum_vals = sorted({v for v in vals if v})
            if 'None' not in enum_vals:
                enum_vals.append('None')
            attr_props[attr] = {'type': 'string', 'enum': enum_vals}
    
    return {
        'type': 'object',
        'properties': {
            'items': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'item_preview': {'type': 'string'},
                        'attributes': {
                            'type': 'object',
                            'properties': attr_props,
                            'required': list(attr_props.keys()),
                            'additionalProperties': False,
                            'propertyOrdering': list(attr_props.keys())
                        }
                    },
                    'required': ['item_preview', 'attributes'],
                    'additionalProperties': False,
                    'propertyOrdering': ['item_preview', 'attributes']
                }
            }
        },
        'required': ['items'],
        'additionalProperties': False,
        'propertyOrdering': ['items']
    }


def build_batch_prompt(hs4: str, product: str, attr_values: Dict[str, List[str]], goods_list: List[str], batch_num: int, total_batches: int, attr_definitions: Dict[str, str], config: Dict[str, Any]) -> str:
    """Build classification prompt for a batch of goods."""
    max_preview = config.get('max_hints_preview', 25)
    
    lines: List[str] = []
    lines.append("TASK: For each goods description pick exactly one allowed value per attribute, or 'None' if nothing clearly fits. You MUST NOT invent new values.")
    lines.append(f"HS4 {hs4} | Product {product} | Batch {batch_num}/{total_batches}")
    
    if attr_definitions:
        lines.append("\nATTRIBUTE DEFINITIONS:")
        for attr in attr_values.keys():
            attr_key = f"attr_{attr}" if not attr.startswith('attr_') else attr
            definition = attr_definitions.get(attr_key, "")
            if definition:
                lines.append(f"- {attr}: {definition}")
            else:
                lines.append(f"- {attr}: [Definition not available]")
    
    lines.append("\nALLOWED VALUE PREVIEW:")
    for attr, vals in attr_values.items():
        pv = ", ".join(vals[:max_preview]) + (" ..." if len(vals) > max_preview else "")
        lines.append(f"- {attr}: {pv}")
    
    lines.append("""

You are a **Senior Commodity Specialist** with deep experience in international trade, commodity classification, and named-entity extraction from shipment descriptions. Your sole task: **extract attribute values from a single text field `Goods shipped` and map them to the provided taxonomy**.

---

# **Step-by-step Workflow**:
1. **Carefully read** the entire `Goods shipped` text. It may contain multiple pieces of information, but you must focus only on the attributes listed.
2. **For each attribute**, identify if any of the allowed values are clearly mentioned in the text. You must pick exactly one value per attribute.
3. **Prioritize exact matches** of the allowed values. If multiple allowed values could match, prefer (in order):
   a. exact token match (whole-word)
   b. longest / most specific textual match
   c. contextual semantic match — choose the most industry-standard option
4. **Use "None" only when completely missing**: If the attribute is not present or cannot be inferred or matched, then and only then use "None".
---

## CRITICAL GUARDRAILS (Violations will cause system errors)

1. **NO EXPLANATIONS:** Never include reasoning, "because", "therefore", or notes in the value. Return ONLY the value.
   - BAD: "Can, the product is in can."
   - GOOD: "Can"
2. **NO LISTS:** Never return multiple values. If multiple apply, pick the most specific/important one.
   - BAD: "Dehydrated, Dried, Toasted"
   - GOOD: "Toasted"
3. **STRICT "None":** If the value is missing, return the string "None". Never "None, because..." or "None (missing)".
   - BAD: "None, CEBOLLA EN POLVO"
   - GOOD: "None"
4. **NO HALLUCINATIONS:** Do not invent values that are not in the text. If it's not there, it's "None".
5. **STRICT SCHEMA ADHERENCE:** You MUST pick from the allowed list. Do NOT invent new values. Do NOT output "custom".

## Hard rules (follow exactly)

1. **Normalization:** Ignore case, punctuation, and hyphen/space differences. Treat obvious spelling errors, plural/singular forms, and verb endings as matches when intent is clear. Be language-agnostic.
2. **Prefer allowed values:** You **must** prioritize matching values from the allowed list for each attribute.
3. **Output format:** For each attribute, return:
   * `"value"`: a string from the allowed list
   * or
   * `"None"`: if no value matches.
   Example:
   ```json
   {
     "Form": { "value": "Powder" },
     "Color": { "value": "None" }
   }
   ```
4. **One unique value per attribute:** Return a **single, clear, unique** value per attribute (no arrays, no multiple choices).
5. **Use `"None"` only when completely missing:** If the attribute is not present or cannot be inferred or matched, then and only then use:
   ```json
   "AttributeName": "None"
   ```
6. **Inference only when obvious:** Apply domain-specific inference **only** when the conclusion is unambiguous (e.g., "not ground nor crushed" -> `Form = "Whole"`). Avoid speculative or creative guesses.
7. **Units & minor variants:** Ignore minor unit/abbreviation differences (g/l, gl, gr/l, kg, etc.) when they don't impact the attribute value. Do not convert or normalize numeric quantities unless the taxonomy explicitly requires a numeric value.
8. **Strict output format:** Output must be **ONLY** the JSON block described above. No extra text, no comments, no trailing commas.

---

""")
    
    lines.append("\nGOODS:")
    for i, g in enumerate(goods_list, start=1):
        lines.append(f"{i}. {g}")
    lines.append("\nIMPORTANT: Return results in the SAME ORDER as the input list (1, 2, 3...).")
    lines.append("For each item, include 'item_preview' with the first 30 characters of the goods description as a verification guardrail.")
    lines.append("Do NOT echo the full goods_shipped text - only return item_preview (first ~30 chars) and attributes.")
    lines.append("\nRETURN JSON ONLY.")
    
    return "\n".join(lines)


def heuristic_fill(attrs: Dict[str, str], goods_text: str, attr_values: Dict[str, List[str]], config: Dict[str, Any]) -> Tuple[Dict[str, str], List[str]]:
    """Apply heuristic extraction for missing attributes."""
    if not config.get('enable_heuristic_fill', False):
        return attrs, []
    
    filled: List[str] = []
    norm_text = goods_text.lower()
    # Simple normalization for matching
    import re
    norm_text_simple = re.sub(r'[#,;/\\()]', ' ', norm_text)
    
    for attr, allowed_list in attr_values.items():
        if attr not in attrs or attrs[attr] != 'None':
            continue
        
        candidates = []
        for val in allowed_list:
            lv = val.lower()
            if lv and lv in norm_text_simple:
                start_idx = norm_text_simple.find(lv)
                if start_idx >= 0:
                    end_idx = start_idx + len(lv)
                    left_ok = start_idx == 0 or norm_text_simple[start_idx-1].isspace()
                    right_ok = end_idx == len(norm_text_simple) or norm_text_simple[end_idx:end_idx+1].isspace()
                    if left_ok and right_ok:
                        candidates.append(val)
        
        if candidates:
            chosen = sorted(candidates, key=lambda x: (-len(x), x))[0]
            attrs[attr] = chosen
            filled.append(attr)
    
    return attrs, filled


def classify_batch(hs4: str, product: str, attr_values: Dict[str, List[str]], goods_list: List[str], model, tracker: TokenTracker, batch_num: int, total_batches: int, attr_definitions: Dict[str, str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Classify a batch of goods descriptions."""
    if tracker.remaining() <= 0:
        return []
    
    # Deterministic first pass
    det_stats = {}
    if config.get('deterministic_first_pass'):
        dm = DeterministicMatcher(attr_values, config)
        deterministic_results = dm.classify_goods(goods_list)
        det_stats = dm.coverage()
    else:
        deterministic_results = [{'goods_shipped': g, 'attributes': {}} for g in goods_list]
        det_stats = {
            'rows_with_any_match': 0,
            'rows_total': len(goods_list),
            'attribute_matches': 0,
            'attribute_attempted': len(goods_list) * len(attr_values),
            'attribute_match_rate': 0.0,
            'negation_blocked': 0
        }
    
    full_attr_count = len(attr_values)
    all_resolved = config.get('deterministic_first_pass') and all(len(r['attributes']) == full_attr_count for r in deterministic_results)
    
    if config.get('dry_run_deterministic_only'):
        all_resolved = True
    
    if all_resolved:
        out = []
        for rec in deterministic_results:
            comp = {a: rec['attributes'].get(a, 'None') for a in attr_values.keys()}
            out.append({
                'goods_shipped': rec['goods_shipped'],
                'attributes': comp,
                '_validation_meta': {
                    'invalid': 0,
                    'total': len(attr_values),
                    'deterministic_only': True,
                    'deterministic_stats': det_stats
                }
            })
        return out
    
    # LLM classification
    response_schema = build_multiitem_response_schema(attr_values, config.get('allow_out_of_schema_values', True))
    prompt = build_batch_prompt(hs4, product, attr_values, goods_list, batch_num, total_batches, attr_definitions, config)
    
    retries = 0
    supports_schema = config.get('use_structured_output', True)
    
    while retries <= config.get('max_retries', 3):
        try:
            # LLMClient handles schema and temperature
            response = model.generate(
                prompt, 
                schema=response_schema if supports_schema else None,
                temperature=config.get('temperature', 0.0)
            )
            
            text = getattr(response, 'text', '')
            usage = getattr(response, 'usage_metadata', None)
            
            used_tokens = 0
            if usage:
                try:
                    used_tokens = int(getattr(usage, 'prompt_token_count', 0)) + int(getattr(usage, 'candidates_token_count', 0))
                except Exception:
                    used_tokens = 0
            if used_tokens == 0:
                used_tokens = (len(prompt) + len(text)) // 4
            
            tracker.add(used_tokens)
            
            parsed = parse_json_response(text)
            out: List[Dict[str, Any]] = []
            invalid_counter = 0
            total_assignments = 0
            out_of_schema_counter = 0
            
            if isinstance(parsed, dict) and isinstance(parsed.get('items'), list):
                # Use positional matching - response order must match input order
                for i, item in enumerate(parsed['items']):
                    if not isinstance(item, dict):
                        continue
                    
                    # Get the corresponding goods description by position
                    if i >= len(goods_list):
                        logger.warning(f"Response has more items than input ({i+1} > {len(goods_list)})")
                        break
                    
                    gs = goods_list[i]  # Use input goods by position
                    
                    # Verify preview as guardrail (optional - log mismatch but continue)
                    item_preview = item.get('item_preview', '')
                    if item_preview:
                        expected_preview = gs[:30]
                        if not expected_preview.startswith(item_preview[:15]):
                            logger.debug(f"Preview mismatch at position {i}: expected '{expected_preview[:20]}...' got '{item_preview[:20]}...'")
                    
                    attrs_in = item.get('attributes', {})
                    
                    norm_attrs: Dict[str, str] = {}
                    raw_attrs: Dict[str, Any] = {}
                    custom_flags: List[str] = []
                    
                    for attr in attr_values.keys():
                        total_assignments += 1
                        raw_v = None
                        if isinstance(attrs_in, dict):
                            raw_v = attrs_in.get(attr, 'None')
                        val = raw_v
                        was_custom_nested = False
                        
                        if isinstance(val, dict):
                            if 'value' in val and isinstance(val['value'], str):
                                val = val['value']
                            elif 'custom' in val and isinstance(val['custom'], str):
                                val = val['custom']
                                was_custom_nested = True
                                if val not in attr_values[attr] and val != 'None':
                                    out_of_schema_counter += 1
                                    custom_flags.append(attr)
                            else:
                                val = 'None'
                        
                        if not isinstance(val, str):
                            val = 'None' if val is None else str(val)
                        
                        # Validate value length to prevent hallucinated explanations
                        if len(val) > 60:
                            # Treat long values as hallucinations and retry the batch
                            msg = f"Hallucination detected: value length {len(val)} > 60 for {attr}"
                            if config.get('log_invalid_values', True):
                                logger.warning(msg)
                            raise ValueError(msg)
                        
                        if not was_custom_nested and val not in attr_values[attr] and val != 'None':
                            if config.get('allow_out_of_schema_values'):
                                out_of_schema_counter += 1
                                custom_flags.append(attr)
                            else:
                                invalid_counter += 1
                                if config.get('log_invalid_values'):
                                    logger.debug(f"Invalid value coerced -> hs4={hs4} product={product} attr={attr} raw='{val}'")
                                val = 'None'
                        
                        norm_attrs[attr] = val
                        if config.get('record_raw_values'):
                            raw_attrs[attr] = raw_v
                    
                    # Merge deterministic assignments
                    if config.get('deterministic_first_pass'):
                        det_lookup = next((d for d in deterministic_results if d['goods_shipped'] == gs), None)
                        if det_lookup:
                            for a, v_det in det_lookup['attributes'].items():
                                if v_det and v_det in attr_values.get(a, []):
                                    if norm_attrs.get(a) != v_det:
                                        norm_attrs[a] = v_det
                                        if a in custom_flags:
                                            custom_flags.remove(a)
                    
                    record: Dict[str, Any] = {'goods_shipped': gs, 'attributes': norm_attrs}
                    if custom_flags:
                        record['custom_attributes'] = custom_flags
                    if config.get('record_raw_values'):
                        record['raw_attributes'] = raw_attrs
                    out.append(record)
            
            if not out:
                for gs in goods_list:
                    for _attr in attr_values.keys():
                        total_assignments += 1
                    out.append({'goods_shipped': gs, 'attributes': {a: 'None' for a in attr_values.keys()}})
            
            if out:
                out[0]['_validation_meta'] = {
                    'invalid': invalid_counter,
                    'total': total_assignments,
                    'deterministic_stats': det_stats,
                    'out_of_schema': out_of_schema_counter if config.get('allow_out_of_schema_values') else 0
                }
            
            # Heuristic enrichment
            heuristic_total_fills = 0
            for rec in out:
                rec_attrs, filled = heuristic_fill(rec['attributes'], rec['goods_shipped'], attr_values, config)
                if filled:
                    heuristic_total_fills += len(filled)
                    rec['attributes'] = rec_attrs
                    if config.get('record_raw_values'):
                        rec.setdefault('heuristic_filled', filled)
            
            if out:
                out[0]['_validation_meta']['heuristic_fills'] = heuristic_total_fills
            
            return out
        
        except TypeError:
            if supports_schema:
                logger.info('Structured schema unsupported; retrying without schema.')
                supports_schema = False
                retries += 1
                time.sleep(config.get('retry_delay', 4))
                continue
            retries += 1
            time.sleep(config.get('retry_delay', 4) * retries)
        except Exception as e:
            msg = str(e)
            if supports_schema and ('response_schema' in msg or 'Unknown field' in msg):
                logger.info('API rejected response_schema; retrying without schema.')
                supports_schema = False
                retries += 1
                time.sleep(config.get('retry_delay', 4))
                continue
            retries += 1
            logger.warning(f"Retry {retries} failed for batch: {e}")
            time.sleep(config.get('retry_delay', 4) * retries)
    
    return []
