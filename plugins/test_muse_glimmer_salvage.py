import importlib.util
import logging

logging.basicConfig(level=logging.WARNING)
spec = importlib.util.spec_from_file_location("muse_glimmer_salvage_parser", "/tmp/muse_glimmer_salvage_parser.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

print("registry has salvage:", "muse_glimmer_salvage" in ToolParserManager.tool_parsers,
      "| stock:", "muse_glimmer" in ToolParserManager.tool_parsers or "muse_glimmer" in ToolParserManager.lazy_parsers)
P = mod.MuseGlimmerSalvageToolParser


class Req:  # duck-typed request with OpenAI-style tool dicts
    tools = [
        {"type": "function", "function": {"name": "terminal", "parameters": {
            "type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
            "required": ["command"]}}},
        {"type": "function", "function": {"name": "memory", "parameters": {
            "type": "object", "properties": {"action": {"type": "string"}, "target": {"type": "string"},
                                             "content": {"type": "string"}},
            "required": ["target"]}}},
        {"type": "function", "function": {"name": "get_weather", "parameters": {
            "type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    ]


reg = P._registered_names(Req)
cases = {
    "dotted+bare (the 2.7% defect) -> terminal/command":
        ' to=self<|message|>x<|eom|><|start|>assistant to=terminal.command<|message|><atem:function_calls>\n'
        '<atem:invoke name="terminal.command">wc -l /opt/data/SOUL.md</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eot|>',
    "dotted tail NOT a property -> single required (command)":
        ' to=terminal.run<|message|><atem:function_calls>\n<atem:invoke name="terminal.run">ls</atem:invoke>\n</atem:function_calls><|eot|>',
    "clean, untouched":
        ' to=self<|message|>x<|eom|><|start|>assistant to=terminal<|message|><atem:function_calls>\n<atem:invoke name="terminal">\n'
        '<atem:parameter name="command">ls -la</atem:parameter>\n<atem:parameter name="timeout">30</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eot|>',
    "x.x stock rule, silent":
        ' to=get_weather.get_weather<|message|><atem:function_calls>\n<atem:invoke name="get_weather.get_weather">\n'
        '<atem:parameter name="location">Oslo</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eot|>',
    "memory.action bare body -> memory/action (tail names the property)":
        ' to=memory.action<|message|><atem:function_calls>\n<atem:invoke name="memory.action">save this</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eot|>',
    "dotted, params present -> name fixed only":
        ' to=terminal.command<|message|><atem:function_calls>\n<atem:invoke name="terminal.command">\n'
        '<atem:parameter name="command">pwd</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eot|>',
    "unknown tool untouched":
        ' to=tweet<|message|><atem:function_calls>\n<atem:invoke name="tweet">\n<atem:parameter name="text">hi</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eot|>',
    "empty body, no params -> {} as before":
        ' to=terminal.command<|message|><atem:function_calls>\n<atem:invoke name="terminal.command">\n</atem:invoke>\n</atem:function_calls><|eot|>',
    "mangle (<|message|> mid-invoke) still no call":
        ' to=self<|message|>x<|eom|><|start|>assistant to=terminal<|message|><atem:function_calls>\n<atem:invoke name="terminal<|message|>">\n'
        '<atem:parameter name="command">wc -l f</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eot|>',
    "reasoning echo not parsed":
        ' to=self<|message|>I would call <atem:invoke name="terminal">\n<atem:parameter name="command">rm -rf /</atem:parameter>\n'
        '</atem:invoke><|eom|><|start|>assistant to=user<|message|>no<|eot|>',
}
for label, raw in cases.items():
    calls = P._parse_tool_calls(raw, reg)
    print(f"{label}: {[(c.function.name, c.function.arguments) for c in calls]}")
print("extract/streaming inherited:", P.extract_tool_calls is mod.MuseGlimmerToolParser.extract_tool_calls,
      P.extract_tool_calls_streaming is mod.MuseGlimmerToolParser.extract_tool_calls_streaming)
