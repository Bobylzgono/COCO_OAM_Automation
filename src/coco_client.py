"""Client for the public COCO Y0 online engine.

The COCO engine is hosted at:
  https://miau.my-x.hu/myx-free/coco/beker_y0.php

It is a legacy HTML form. Because the form field names are not guaranteed to be
stable, this client *discovers* the form structure at runtime:

- GET the page
- Parse the first <form>
- Pick the textarea that corresponds to the required "Mátrix" field
- Fill optional object/attribute name lists if present
- Set model to Y0 if the form has a model selector
- POST to the form action and return the resulting HTML

If Streamlit Cloud blocks outbound requests, run this locally or host a local
copy of the COCO engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


COCO_URL = "https://miau.my-x.hu/myx-free/coco/beker_y0.php"


@dataclass
class CocoForm:
    url: str
    action_url: str
    fields: Dict[str, str]
    matrix_field_name: str
    object_list_field_name: Optional[str]
    attribute_list_field_name: Optional[str]
    textarea_names: list[str]
    submit_field_name: Optional[str]
    submit_field_value: str


def _best_textarea_for_matrix(form) -> str:
    """Heuristic: choose textarea whose name looks like matrix or is the only textarea."""
    textareas = form.find_all("textarea")
    if not textareas:
        raise RuntimeError("COCO page has no <textarea>; cannot locate matrix field.")

    if len(textareas) == 1:
        return textareas[0].get("name")

    # Prefer names that look like matrix
    candidates = []
    for ta in textareas:
        name = (ta.get("name") or "").lower()
        score = 0
        if "mat" in name or "matrix" in name:
            score += 10
        if "rang" in name:
            score += 5
        # bigger textarea likely matrix
        try:
            score += int(ta.get("cols") or 0) // 10
            score += int(ta.get("rows") or 0) // 10
        except Exception:
            pass
        candidates.append((score, ta.get("name"), ta))

    candidates.sort(reverse=True, key=lambda x: x[0])
    best = candidates[0][1]
    if not best:
        # fallback to first textarea
        best = textareas[0].get("name")
    if not best:
        raise RuntimeError("Could not determine name attribute for matrix textarea.")
    return best


def _guess_optional_list_field(form, kind: str) -> Optional[str]:
    """Try to locate optional fields for object or attribute names."""
    # kind in {'object','attribute'}
    keys = {
        "object": ["obj", "object", "rekord", "record", "sor", "row", "nev", "name"],
        "attribute": ["attr", "attrib", "tulajd", "feature", "variable", "oszlop", "col", "jellemz"],
    }[kind]

    def _score_textarea(ta) -> int:
        score = 0
        name = (ta.get("name") or "").lower()
        if any(k in name for k in keys):
            score += 10
        near = " ".join(
            [
                str(ta.get("id") or ""),
                str(ta.parent.get_text(" ", strip=True) if ta.parent else ""),
                str(ta.find_previous(string=True) or ""),
            ]
        ).lower()
        if any(k in near for k in keys):
            score += 4
        return score

    best_name = None
    best_score = 0
    for ta in form.find_all("textarea"):
        name = ta.get("name")
        if not name:
            continue
        score = _score_textarea(ta)
        if score > best_score:
            best_score = score
            best_name = name
    if best_name and best_score > 0:
        return best_name

    # Some forms use input fields (rare)
    for inp in form.find_all("input"):
        name = (inp.get("name") or "").lower()
        if any(k in name for k in keys):
            return inp.get("name")

    return None


def discover_coco_form(url: str = COCO_URL, timeout: int = 30) -> CocoForm:
    """Download the COCO page and discover form field names."""
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    form = soup.find("form")
    if not form:
        raise RuntimeError("COCO page does not contain a <form>.")

    action = form.get("action") or url
    action_url = urljoin(url, action)

    # Collect default values for all fields (hidden inputs, etc.)
    fields: Dict[str, str] = {}
    submit_field_name: Optional[str] = None
    submit_field_value: str = ""
    for inp in form.find_all("input"):
        name = inp.get("name")
        if not name:
            continue
        typ = (inp.get("type") or "").lower()
        if typ in {"submit", "button"}:
            if submit_field_name is None:
                submit_field_name = name
                submit_field_value = inp.get("value") or "1"
            continue
        if typ in {"image", "file"}:
            continue
        fields[name] = inp.get("value") or ""

    # Select / option defaults
    for sel in form.find_all("select"):
        name = sel.get("name")
        if not name:
            continue
        opt = sel.find("option", selected=True) or sel.find("option")
        if opt is not None:
            fields[name] = opt.get("value") or opt.text

    # Textarea defaults
    textarea_names: list[str] = []
    for ta in form.find_all("textarea"):
        name = ta.get("name")
        if not name:
            continue
        textarea_names.append(name)
        fields.setdefault(name, ta.text or "")

    matrix_field_name = _best_textarea_for_matrix(form)
    object_list_field_name = _guess_optional_list_field(form, "object")
    attribute_list_field_name = _guess_optional_list_field(form, "attribute")

    return CocoForm(
        url=url,
        action_url=action_url,
        fields=fields,
        matrix_field_name=matrix_field_name,
        object_list_field_name=object_list_field_name,
        attribute_list_field_name=attribute_list_field_name,
        textarea_names=textarea_names,
        submit_field_name=submit_field_name,
        submit_field_value=submit_field_value,
    )


def _set_model_y0(fields: Dict[str, str]) -> None:
    """Try to force model to Y0 (exact field name varies)."""
    # Common candidates seen in legacy PHP forms
    for k in list(fields.keys()):
        kl = k.lower()
        if kl in {"modell", "model", "mod"}:
            fields[k] = "Y0"

    # Sometimes radio buttons / hidden field uses 'y0' / 'y_0'
    for k in list(fields.keys()):
        if "y0" in k.lower():
            fields[k] = fields[k] or "1"


def run_coco_y0(
    matrix_tsv: str,
    object_names: Optional[Sequence[str] | str] = None,
    attribute_names: Optional[Sequence[str] | str] = None,
    steps: Optional[int] = None,
    identifier: str = "Teszt",
    timeout: int = 60,
) -> Tuple[str, CocoForm]:
    """Submit a ranked matrix to COCO Y0 and return (html, discovered_form)."""
    form = discover_coco_form(timeout=timeout)
    payload = dict(form.fields)

    _set_model_y0(payload)

    # Identifier field (best effort)
    for k in list(payload.keys()):
        if k.lower() in {"azonosito", "id", "identifier"}:
            payload[k] = identifier

    # Steps (stairs) field
    if steps is not None:
        for k in list(payload.keys()):
            if any(s in k.lower() for s in ["lepcso", "stairs", "stair", "step"]):
                payload[k] = str(int(steps))

    # Prefer canonical field names when present on this legacy form.
    matrix_field = "matrix" if "matrix" in form.fields else form.matrix_field_name
    object_field = "object" if "object" in form.fields else form.object_list_field_name
    attribute_field = "attribute" if "attribute" in form.fields else form.attribute_list_field_name

    # Required matrix. Normalize line breaks for legacy parser stability.
    matrix_norm = matrix_tsv.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\r\n")
    payload[matrix_field] = matrix_norm

    # Fallback: if optional list fields were not discovered, map remaining textareas.
    remaining_textareas = [n for n in form.textarea_names if n != form.matrix_field_name]
    if object_names and not object_field and remaining_textareas:
        object_field = remaining_textareas[0]
    if attribute_names and not attribute_field:
        for name in remaining_textareas:
            if name != object_field:
                attribute_field = name
                break

    def _to_block(value: Optional[Sequence[str] | str], sep: str) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return sep.join(str(v) for v in value)

    # Optional labels
    object_block = _to_block(object_names, "\n")
    attribute_block = _to_block(attribute_names, "\n")
    if object_block is not None:
        object_block = object_block.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\r\n")
    if attribute_block is not None:
        attribute_block = attribute_block.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\r\n")
    if object_block and object_field:
        payload[object_field] = object_block
    if attribute_block and attribute_field:
        payload[attribute_field] = attribute_block

    # Some legacy handlers only execute when submit field is present.
    if form.submit_field_name:
        payload[form.submit_field_name] = form.submit_field_value or "1"

    resp = requests.post(form.action_url, data=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.text, form
