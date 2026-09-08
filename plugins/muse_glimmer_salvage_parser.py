# SPDX-License-Identifier: Apache-2.0
"""muse_glimmer_salvage — vLLM tool-parser plugin: MuseGlimmerToolParser plus two
CONSERVATIVE salvages for measured Muse emission defects (2026-09-08, see
posttrain/muse/ATEM-LANE.md; store: 30/1110 Muse tool calls = 2.7%).

 1. dotted-name: `<atem:invoke name="terminal.command">` where `terminal` IS a
    registered tool and `terminal.command` is NOT -> `terminal`. The stock parser
    only collapses `x.x`; every other dotted name passes through and Hermes
    answers "Tool 'terminal.command' does not exist" (x58 in 60 days).
 2. bare-body parameter: the same emissions carry the argument as bare text —
    `...name="terminal.command">wc -l f</atem:parameter>` — with NO
    `<atem:parameter name=...>` open tag, so the stock parser returns arguments
    `{}` and the command text is lost. When an invoke parsed ZERO parameters, the
    body is non-empty, and the (salvaged) tool has exactly one required parameter
    (or no required and exactly one property), bind the body to it. String-typed
    parameters keep the raw text; others go through the parser's JSON decode.

Both log at WARNING "ATEM salvage: ..." so the rate stays countable in the vLLM
log (`docker logs vllm_node 2>&1 | grep -c "ATEM salvage"`). Anything still
unrecognisable is passed through exactly as before — Hermes' unknown-tool error
stays the net. The raw classifier in posttrain/muse/bin/atem_common.py stays
STRICT on purpose (weights-level rate); the served parser is what prod sees.

Serve: --tool-parser-plugin <this file> --tool-call-parser muse_glimmer_salvage
Lives on the host under the HF cache (mounted into every recipe container);
keep next to the served vLLM version — it subclasses that image's parser.
"""
from __future__ import annotations

import json
import re

from vllm.entrypoints.generate.base.protocol import FunctionCall, ToolCall
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
from vllm.tool_parsers.muse_glimmer_tool_parser import (
    _INVOKE_RE,
    _NAME_RE,
    _PARAM_RE,
    MuseGlimmerToolParser,
    _decode_value,
    logger,
)

_BODY_RE = re.compile(r"<atem:invoke\b[^>]*>(?P<body>.*)</atem:invoke>", re.DOTALL)
_DANGLING_CLOSE_RE = re.compile(r"\s*</atem:parameter>\s*$")


class _Registered(set):
    """Registered tool names, carrying each tool's JSON-schema parameters."""

    schemas: dict


@ToolParserManager.register_module(["muse_glimmer_salvage"])
class MuseGlimmerSalvageToolParser(MuseGlimmerToolParser):
    @staticmethod
    def _registered_names(request):
        names = _Registered()
        names.schemas = {}
        tools = getattr(request, "tools", None) if request is not None else None
        for t in tools or []:
            # pydantic ChatCompletionToolsParam (.function.name) on the wire; plain
            # OpenAI dicts ({"function": {...}}) from tests and offline callers.
            fn = (t.get("function") or t) if isinstance(t, dict) else (getattr(t, "function", None) or t)
            if isinstance(fn, dict):
                name, params = fn.get("name"), fn.get("parameters")
            else:
                name, params = getattr(fn, "name", None), getattr(fn, "parameters", None)
            if name:
                names.add(name)
                names.schemas[name] = params if isinstance(params, dict) else {}
        return names

    @staticmethod
    def _body_param(schema, tail):
        """Which parameter a bare invoke body belongs to, most faithful reading first:
        the dotted tail when it names a property (`terminal.command` -> command),
        else the tool's single required parameter, else its only property."""
        props = (schema or {}).get("properties") or {}
        req = (schema or {}).get("required") or []
        if tail and tail in props:
            return tail, (props[tail] or {}).get("type")
        if len(req) == 1:
            return req[0], (props.get(req[0]) or {}).get("type")
        if not req and len(props) == 1:
            k = next(iter(props))
            return k, (props[k] or {}).get("type")
        return None, None

    @classmethod
    def _parse_tool_calls(cls, text, registered=None):
        registered = registered or set()
        schemas = getattr(registered, "schemas", None) or {}
        scoped = cls._tool_channel_text(text)
        tool_calls: list[ToolCall] = []
        for invoke in _INVOKE_RE.findall(scoped):
            name_m = _NAME_RE.search(invoke)
            if not name_m:
                continue
            emitted = name_m.group(1)
            tail = None
            if not registered or emitted in registered:
                name = emitted
            else:
                head, sep, tail = emitted.partition(".")
                if sep and head in registered:
                    name = head  # covers the stock x.x rule and the dotted-name defect
                    if tail != head:
                        logger.warning("ATEM salvage: dotted tool name %r -> %r", emitted, name)
                else:
                    name = cls._normalize_name(emitted, registered)  # stock pass-through + warning
                    tail = None
            args: dict = {}
            for pm in _PARAM_RE.finditer(invoke):
                args[pm.group("key")] = _decode_value(pm.group("value"))
            if not args:
                body_m = _BODY_RE.search(invoke)
                body = _DANGLING_CLOSE_RE.sub("", body_m.group("body")).strip() if body_m else ""
                param, ptype = cls._body_param(schemas.get(name), tail)
                if body and param:
                    args[param] = body if ptype == "string" else _decode_value(body)
                    logger.warning("ATEM salvage: bare invoke body bound to %s.%s (%d chars)",
                                   name, param, len(body))
            tool_calls.append(
                ToolCall(function=FunctionCall(name=name, arguments=json.dumps(args, ensure_ascii=False)))
            )
        return tool_calls


# The decorator above only records a LAZY (module_path, class_name) mapping that
# get_tool_parser() re-imports by module name; register eagerly as well so the
# name resolves regardless of how the plugin file was imported.
ToolParserManager.register_module(name="muse_glimmer_salvage", module=MuseGlimmerSalvageToolParser)
