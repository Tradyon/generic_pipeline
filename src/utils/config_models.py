"""Shared configuration models for the generic pipeline.

These helpers provide a single place where product definitions, attribute
schemas, and attribute definitions are parsed, validated at a basic level, and
serialized back to JSON. They exist so that both manually curated configs and
auto-generated configs (via ``schema_generator``) share identical structures.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Product definition
# ---------------------------------------------------------------------------


@dataclass
class ProductDefinition:
    hs_code: str
    product_categories: List[str]
    category_descriptions: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "hs_code": self.hs_code,
            "product_categories": list(self.product_categories),
            "category_descriptions": dict(self.category_descriptions),
        }
        payload.update(self.metadata)
        return payload


def load_product_definition(path: str) -> ProductDefinition:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Product definition file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    hs_code = str(raw.get("hs_code", "")).strip()
    categories = raw.get("product_categories", []) or []
    descriptions = raw.get("category_descriptions", {}) or {}
    metadata = {
        k: v
        for k, v in raw.items()
        if k not in {"hs_code", "product_categories", "category_descriptions"}
    }

    return ProductDefinition(
        hs_code=hs_code,
        product_categories=[str(c).strip() for c in categories if str(c).strip()],
        category_descriptions={str(k).strip(): str(v).strip() for k, v in descriptions.items()},
        metadata=metadata,
    )


def dump_product_definition(definition: ProductDefinition, path: str) -> None:
    payload = definition.to_dict()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Attribute schema
# ---------------------------------------------------------------------------


@dataclass
class AttributeSet:
    attributes: Dict[str, List[str]] = field(default_factory=dict)

    def normalized_values(self) -> Dict[str, List[str]]:
        return {k: list(v) for k, v in self.attributes.items()}


@dataclass
class HSAttributeSchema:
    version: str
    products: Dict[str, AttributeSet] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "version": self.version,
            "products": {
                product: {"attributes": aset.normalized_values()}
                for product, aset in self.products.items()
            },
        }
        payload.update(self.metadata)
        return payload


@dataclass
class AttributeSchema:
    entries: Dict[str, HSAttributeSchema] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {hs: entry.to_dict() for hs, entry in self.entries.items()}
        for key, value in self.metadata.items():
            if key not in payload:
                payload[key] = value
        return payload

    def to_normalized_nested_map(self) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
        """Return mapping used by attribute classifiers."""
        result: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
        for hs, entry in self.entries.items():
            result[hs] = {
                product: aset.normalized_values()
                for product, aset in entry.products.items()
            }
        return result


def load_attribute_schema(path: str) -> AttributeSchema:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Attribute schema file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    entries: Dict[str, HSAttributeSchema] = {}
    metadata: Dict[str, object] = {}

    for hs_code, block in raw.items():
        if not isinstance(block, dict):
            metadata[hs_code] = block
            continue

        version = str(block.get("version", "")).strip() or datetime.utcnow().isoformat() + "Z"
        products_raw = block.get("products", {}) or {}

        products: Dict[str, AttributeSet] = {}
        for product_name, pdata in products_raw.items():
            if not isinstance(pdata, dict):
                continue
            attr_map = pdata.get("attributes", {}) or {}
            cleaned = {
                str(attr).strip(): [str(v).strip() for v in values if str(v).strip()]
                for attr, values in attr_map.items()
                if isinstance(values, list)
            }
            if cleaned:
                products[str(product_name).strip()] = AttributeSet(cleaned)

        extras = {
            k: v for k, v in block.items() if k not in {"version", "products"}
        }
        entries[str(hs_code).strip()] = HSAttributeSchema(
            version=version,
            products=products,
            metadata=extras,
        )

    return AttributeSchema(entries=entries, metadata=metadata)


def dump_attribute_schema(schema: AttributeSchema, path: str) -> None:
    payload = schema.to_dict()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Attribute definitions
# ---------------------------------------------------------------------------


@dataclass
class AttributeDefinitions:
    definitions: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        payload = dict(self.metadata)
        payload.update({k: v for k, v in self.definitions.items()})
        return payload


def load_attribute_definitions(path: str) -> AttributeDefinitions:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Attribute definitions file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    metadata: Dict[str, object] = {}
    definitions: Dict[str, str] = {}

    for key, value in raw.items():
        if key.startswith("attr_"):
            definitions[key] = str(value)
        else:
            metadata[key] = value

    return AttributeDefinitions(definitions=definitions, metadata=metadata)


def dump_attribute_definitions(definitions: AttributeDefinitions, path: str) -> None:
    payload = definitions.to_dict()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


# Convenience helpers -------------------------------------------------------


def ensure_output_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


