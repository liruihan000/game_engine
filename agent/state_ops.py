from typing import Any, Dict, List, Tuple


def _compute_start_counter(items: List[Dict[str, Any]], items_created: int) -> int:
    """Derive the starting numeric counter from itemsCreated and max existing numeric id."""
    prior = items_created or 0
    max_existing = 0
    for it in items:
        try:
            parsed = int(str(it.get("id", "0")), 10)
            if parsed > max_existing:
                max_existing = parsed
        except Exception:
            continue
    return max(prior, max_existing)


def _find_existing_by_type_and_name(items: List[Dict[str, Any]], item_type: str, name: str) -> Dict[str, Any] | None:
    normalized = (name or "").strip()
    if not normalized:
        return None
    for it in items:
        if it.get("type") == item_type and str(it.get("name", "")).strip() == normalized:
            return it
    return None


def _normalize_item_id_for_match(item_id: str) -> List[str]:
    """
    Normalize a provided item id for matching:
    - Keep the original as-is
    - If it's purely numeric, also include its 4-digit zero-padded form
    This mirrors frontend behavior while being more tolerant of model outputs like "0", "1", etc.
    """
    candidates = [str(item_id)]
    try:
        if str(item_id).isdigit():
            n = int(str(item_id), 10)
            candidates.append(str(n).zfill(4))
    except Exception:
        pass
    return list(dict.fromkeys(candidates))  # de-duplicate preserving order


def add_item(
    items: List[Dict[str, Any]],
    items_created: int,
    item_type: str,
    name: str,
    data: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], int, str | None, str | None]:
    """
    Add an item to the list with a stable increasing id. Name-based idempotency.

    Returns: (next_items, next_counter, created_id_or_existing, existing_id_if_skipped)
    - If an item with same (type, name) exists, next_items/counter unchanged, returns (existing_id, existing_id).
    - Otherwise appends new item and returns (new_id, None).
    """
    existing = _find_existing_by_type_and_name(items, item_type, name)
    if existing is not None:
        return items, _compute_start_counter(items, items_created), existing.get("id"), existing.get("id")

    counter = _compute_start_counter(items, items_created) + 1
    new_id = str(counter).zfill(4)
    item: Dict[str, Any] = {
        "id": new_id,
        "type": item_type,
        "name": (name or "").strip(),
        "subtitle": "",
        "data": data,
    }
    return [*items, item], counter, new_id, None


def update_item(
    items: List[Dict[str, Any]],
    item_id: str,
    updates: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], bool]:
    """Shallow-merge updates into the top-level item fields."""
    changed = False
    next_items: List[Dict[str, Any]] = []
    for it in items:
        if str(it.get("id")) == str(item_id):
            merged = {**it, **updates}
            next_items.append(merged)
            changed = True
        else:
            next_items.append(it)
    return next_items, changed


def update_item_data_merge(
    items: List[Dict[str, Any]],
    item_id: str,
    data_patch: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], bool]:
    """Merge a dict patch into item.data."""
    changed = False
    next_items: List[Dict[str, Any]] = []
    for it in items:
        if str(it.get("id")) == str(item_id):
            data = dict(it.get("data", {}))
            data.update(data_patch or {})
            merged = {**it, "data": data}
            next_items.append(merged)
            changed = True
        else:
            next_items.append(it)
    return next_items, changed


def delete_item(
    items: List[Dict[str, Any]],
    item_id: str,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Remove an item by id. Returns (next_items, existed).
    Accepts numeric ids like "1" and treats them as zero-padded "0001" as well.
    """
    match_ids = _normalize_item_id_for_match(item_id)
    existed = any(str(it.get("id")) in match_ids for it in items)
    next_items = [it for it in items if str(it.get("id")) not in match_ids]
    return next_items, existed


