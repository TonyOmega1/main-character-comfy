"""Static checks of an API-format graph against a ComfyUI ``/object_info`` dump.

Used by the unit tests (against a snapshot of the reference machine) and by
``python -m main_character_comfy.doctor`` (against your live ComfyUI) to catch missing
custom nodes, missing model files, bad links and invalid combo values before a
long render is queued.
"""

from __future__ import annotations

from typing import Any


def _is_dynamic_combo(spec: Any) -> bool:
    return isinstance(spec, (list, tuple)) and bool(spec) and spec[0] == "COMFY_DYNAMICCOMBO_V3"


def _dynamic_children(spec: Any, value: Any) -> dict[str, Any] | None:
    """Sub-inputs of a V3 DynamicCombo for the chosen option (``None`` when the option isn't found)."""
    opts = spec[1].get("options", []) if len(spec) > 1 and isinstance(spec[1], dict) else []
    for o in opts:
        if isinstance(o, dict) and o.get("key") == value:
            ins = o.get("inputs") or {}
            return {**(ins.get("required") or {}), **(ins.get("optional") or {})}
    return None


def _undotted_hint(name: str, specs: dict[str, Any]) -> str | None:
    """'mode.mask_1' when ``name`` is a sub-input of a DynamicCombo that was sent without its prefix."""
    for parent, spec in specs.items():
        if not _is_dynamic_combo(spec):
            continue
        for o in (spec[1].get("options", []) if len(spec) > 1 and isinstance(spec[1], dict) else []):
            ins = (o or {}).get("inputs") or {}
            if name in {**(ins.get("required") or {}), **(ins.get("optional") or {})}:
                return f"{parent}.{name}"
    return None


def _combo_options(spec: Any) -> list[Any] | None:
    if not isinstance(spec, (list, tuple)) or not spec:
        return None
    head = spec[0]
    if isinstance(head, list):  # legacy: [[opt1, opt2], {...}]
        return head
    if head == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1].get("options")
    return None


def validate_graph(graph: dict[str, Any], object_info: dict[str, Any], *, check_files: bool = True) -> list[str]:
    """Return a list of human-readable problems (empty list = looks runnable)."""
    problems: list[str] = []
    for nid, node in graph.items():
        ct = node.get("class_type")
        info = object_info.get(ct)
        if info is None:
            problems.append(f"node {nid}: unknown node type '{ct}' (custom node missing?)")
            continue
        inputs = node.get("inputs", {})
        req = (info.get("input") or {}).get("required", {}) or {}
        opt = (info.get("input") or {}).get("optional", {}) or {}
        n_outputs = len(info.get("output") or [])
        for name, spec in req.items():
            if name not in inputs:
                # V3 Autogrow / DynamicCombo parents are sent only as their dotted children ("images.image0")
                if any(k.startswith(f"{name}.") for k in inputs):
                    continue
                problems.append(f"node {nid} ({ct}): missing required input '{name}'")
        for name, value in inputs.items():
            spec = req.get(name, opt.get(name))
            if spec is None and "." in name:  # V3 sub-input "parent.child": check it against the chosen option
                parent, child = name.split(".", 1)
                pspec = req.get(parent, opt.get(parent))
                if _is_dynamic_combo(pspec) and not isinstance(inputs.get(parent), list):
                    kids = _dynamic_children(pspec, inputs.get(parent))
                    if kids is not None and child.split(".", 1)[0] not in kids:
                        problems.append(f"node {nid} ({ct}): '{name}' is not an input of {parent}="
                                        f"'{inputs.get(parent)}'")
                        continue
                    spec = (kids or {}).get(child)
            elif spec is None:
                hint = _undotted_hint(name, {**req, **opt})
                if hint:  # ComfyUI drops it silently: the V3 DynamicCombo only reads the dotted key
                    problems.append(f"node {nid} ({ct}): '{name}' must be sent as '{hint}' (dropped otherwise)")
                    continue
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                src = graph.get(value[0])
                if src is None:
                    problems.append(f"node {nid} ({ct}).{name}: link to missing node {value[0]}")
                    continue
                src_info = object_info.get(src.get("class_type"), {})
                src_outs = len(src_info.get("output") or [])
                if src_info and not (0 <= int(value[1]) < src_outs):
                    problems.append(f"node {nid} ({ct}).{name}: output slot {value[1]} out of range")
                if spec is None and "." not in name:  # a link into an input the node doesn't have: ignored silently
                    problems.append(f"node {nid} ({ct}): unexpected linked input '{name}'")
                continue
            if spec is None:
                if "." in name:  # V3 dynamic-combo sub-input such as "format.codec"
                    continue
                problems.append(f"node {nid} ({ct}): unexpected input '{name}'")
                continue
            options = _combo_options(spec)
            if options is None:
                continue
            is_file_list = name in {
                "unet_name", "clip_name", "vae_name", "lora_name", "name", "audio_encoder_name", "model_name", "ckpt_name",
            }
            if options == [] and is_file_list and check_files:
                problems.append(f"node {nid} ({ct}).{name}: no model files installed for this loader")
                continue
            if options and value not in options:
                if name in {"image", "audio", "file"}:
                    continue  # filled by an uploaded asset at submit time
                if is_file_list and not check_files:
                    continue
                problems.append(f"node {nid} ({ct}).{name}: '{value}' not in available options")
        _ = n_outputs
    return problems
