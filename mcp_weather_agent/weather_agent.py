"""
weather_agent.py - Weather agent: LangChain + LangGraph + LangChain MCP adapters

Flow (matches the assignment):
  1. The user asks a natural-language weather question.
  2. The LLM (Gemma via OpenRouter) identifies the city and calls the
     `geocode` MCP tool (MCP server #1, geocoding_server.py).
  3. The geocoding server returns latitude/longitude.
  4. The LLM calls the `get_weather` MCP tool (MCP server #2, weather_server.py)
     with those coordinates.
  5. The weather server returns current conditions and forecast data.
  6. The tool results are added to the conversation (the prompt context) and the
     LLM writes the final answer to the original question.

Components
  - LangChain:             ChatOpenAI pointed at OpenRouter (OpenAI-compatible API)
  - langchain-mcp-adapters: loads the MCP servers' tools as LangChain tools
  - LangGraph:             StateGraph that loops  agent -> tools -> agent  until
                           the model stops calling tools

Setup:  set the OPENROUTER_API_KEY environment variable (never put the key in code).
Usage:  python weather_agent.py --model <model-id> "What's the weather in Boston?"
        python weather_agent.py            (interactive mode)
        Other OpenAI-compatible providers also work, e.g.
        python weather_agent.py --base-url <url> --key-env <ENV_VAR_NAME> --model <id> "question"
        python weather_agent.py --mode react --model <model-id> "question"   (see "Tool-calling modes")
Every MCP call and response is printed and appended to agent_log.txt (for the report).

Tool-calling modes
  native (default)  the model uses the provider's function-calling API (tools are sent with the request).
                    Works with models that support tool calling (their OpenRouter page lists "tools").
  react             for models WITHOUT reliable native tool calling (e.g. Gemma). The tools are described in
                    the prompt, the model asks for a tool by replying with a JSON line such as
                    {"tool": "geocode", "args": {"city": "Paris"}}, the program parses it and runs the MCP
                    tool, and the tool result is shown to the model as the next message. The LangGraph
                    loop, the MCP servers and the logging are identical in both modes.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
# OpenRouter model id. Pick a FREE model (id ends in ":free") that supports tool
# calling, and check its page on openrouter.ai. Override with --model or the
# OPENROUTER_MODEL environment variable.
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-3-27b-it")
OPENROUTER_URL = "https://openrouter.ai/api/v1"

HERE = Path(__file__).parent
LOG_FILE = HERE / "agent_log.txt"

# Both MCP servers are launched as stdio subprocesses using this Python
# interpreter (so they run inside your virtual environment).
MCP_SERVERS = {
    "geocoding": {
        "command": sys.executable,
        "args": [str(HERE / "geocoding_server.py")],
        "transport": "stdio",
    },
    "weather": {
        "command": sys.executable,
        "args": [str(HERE / "weather_server.py")],
        "transport": "stdio",
    },
}

# Prompt engineering for reliable tool use: explicit step order, no guessing
# coordinates, answer only from tool data.
INSTRUCTIONS = """You are a weather assistant with two tools:
1. geocode(city, country_code) - converts a place name into latitude and longitude.
2. get_weather(latitude, longitude, temperature_unit, forecast_days) - returns current conditions and a daily forecast for coordinates.

Follow these steps for every weather question:
1. Identify the place the user asks about and call geocode first. Never guess coordinates.
2. As soon as geocode succeeds, call get_weather with the latitude and longitude of the best match. If geocode returns several matches, use the first one unless the user named a state or country. Do not answer before you have weather data.
   Use temperature_unit="fahrenheit" for places in the United States unless the user asks otherwise; use "celsius" elsewhere.
3. Answer the user's original question using ONLY the data returned by the tools. Mention the resolved location, the units, and the relevant current or forecast values. Be concise.

If geocode returns an error or no match, say so and ask for a clearer location. If the question is not about weather, answer briefly without using tools."""

# Some hosted Gemma endpoints reject a separate "system" message. By default the
# instructions are therefore placed at the start of the user message, which works
# with every model. Use --system-prompt to send a real system message instead.
# Sent once if the model returns an empty reply (no text and no tool call), which some
# models do right after receiving a tool result.
RETRY_REMINDER = ("Your last reply was empty. Continue the task: if you have coordinates but no "
                  "weather data yet, call get_weather now; if you already have weather data, "
                  "write the final answer.")
LLM_CALLS = 0         # number of LLM requests made (counts toward the daily quota)
MAX_GRAPH_STEPS = 12  # hard cap on graph steps so a looping model cannot burn the daily quota


# --------------------------------------------------------------------------
# Logging helpers (console + agent_log.txt)
# --------------------------------------------------------------------------
def log(text=""):
    """Print a line and append it to the log file."""
    print(text)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def text_of(content):
    """Message content can be a string or a list of content blocks; return plain text."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        parts.append(block.get("text", "") if isinstance(block, dict) else str(block))
    return "".join(parts)


def pretty(text):
    """Pretty-print JSON tool output when possible."""
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except (ValueError, TypeError):
        return text


# --------------------------------------------------------------------------
# Helpers for the "react" mode (tool calls written as JSON text)
# --------------------------------------------------------------------------
def parse_action(text, tool_names):
    """
    Find a tool request in the model's text, e.g.  {"tool": "geocode", "args": {"city": "Paris"}}
    (also accepts a ```json fence or extra words around it). Returns (tool_name, args) or None.
    """
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except ValueError:
            continue
        if isinstance(obj, dict):
            name = obj.get("tool") or obj.get("name")
            args = obj.get("args") or obj.get("arguments") or obj.get("parameters") or {}
            if name in tool_names and isinstance(args, dict):
                return name, args
    return None


def to_plain_messages(messages):
    """
    Convert the LangGraph message history into plain chat messages for a model without native
    tool calling: a tool request becomes the model's own JSON text, and a tool result becomes a
    user message that shows the result.
    """
    plain = []
    for m in messages:
        if isinstance(m, ToolMessage):
            plain.append(HumanMessage(content=(
                f"Tool result from {m.name}:\n{text_of(m.content)}\n\n"
                "Now either call the next tool (reply with the JSON only) or, if you have everything you "
                "need, write the final answer in plain text.")))
        elif isinstance(m, AIMessage) and m.tool_calls:
            call = m.tool_calls[0]
            plain.append(AIMessage(content=json.dumps({"tool": call["name"], "args": call["args"]})))
        else:
            plain.append(m)
    return plain


def react_instructions(tools):
    """Prompt for the react mode: the tools are described in text, plus the JSON calling format."""
    described = []
    for t in tools:
        arguments = ", ".join(f"{k} ({v.get('type', 'any')})" for k, v in t.args.items())
        described.append(f"- {t.name}: {t.description.strip()}\n  Arguments: {arguments}")
    return ("You are a weather assistant. You can call these tools:\n" + "\n".join(described) + "\n\n"
            "To call a tool, reply with ONLY one JSON object and nothing else, in this form:\n"
            '{"tool": "<tool name>", "args": {<arguments>}}\n'
            "The result of the tool is shown to you in the next message.\n\n"
            "Follow these steps for every weather question:\n"
            "1. Identify the place the user asks about and call geocode first. Never guess coordinates.\n"
            "2. As soon as geocode succeeds, call get_weather with the latitude and longitude of the best match. "
            "If geocode returns several matches, use the first one unless the user named a state or country. "
            "Do not answer before you have weather data. Use temperature_unit \"fahrenheit\" for places in the "
            "United States unless the user asks otherwise; use \"celsius\" elsewhere.\n"
            "3. When you have the weather data, answer the user's original question in plain text (not JSON), "
            "using ONLY the data returned by the tools. Mention the resolved location, the units and the relevant "
            "current or forecast values. Be concise.\n"
            "If geocode returns an error or no match, say so in plain text and ask for a clearer location. "
            "If the question is not about weather, answer briefly in plain text without using tools.")


# --------------------------------------------------------------------------
# LangGraph orchestration
# --------------------------------------------------------------------------
def build_graph(llm, tools, mode="native"):
    """
    Build the agent loop as an explicit LangGraph StateGraph:

        START -> agent --(model requested tool calls?)--> tools -> agent -> ...
                   \\--(no tool calls)--> END

    State is a list of messages (MessagesState). The tool results become
    ToolMessages in that list, which is how the retrieved weather data gets
    "integrated into the prompt context" for the final answer.
    mode "native": the model calls tools through the provider's function-calling API.
    mode "react":  the model writes tool requests as JSON text, which are parsed here.
    """
    tool_names = {t.name for t in tools}
    llm_with_tools = llm.bind_tools(tools) if mode == "native" else None

    async def call_model_native(state: MessagesState):
        """Agent node (native mode): the LLM reads all messages so far and replies (maybe with tool calls)."""
        global LLM_CALLS
        messages = state["messages"]
        LLM_CALLS += 1
        response = await llm_with_tools.ainvoke(messages)

        # Robustness: an empty reply (no text, no tool call) would end the run with no
        # answer. Retry once with a short reminder instead.
        if not response.tool_calls and not text_of(response.content).strip():
            log("\n[NOTE] the model returned an empty reply; retrying once with a reminder.")
            log(f"       finish_reason={response.response_metadata.get('finish_reason')!r}")
            nudge = HumanMessage(content=RETRY_REMINDER)
            LLM_CALLS += 1
            retry = await llm_with_tools.ainvoke(messages + [nudge])
            return {"messages": [nudge, retry]}   # the empty reply is not kept in the history
        return {"messages": [response]}

    async def call_model_react(state: MessagesState):
        """Agent node (react mode): plain-text conversation; a JSON reply is turned into a tool call."""
        global LLM_CALLS
        plain = to_plain_messages(state["messages"])
        LLM_CALLS += 1
        text = text_of((await llm.ainvoke(plain)).content).strip()
        if not text:                               # empty reply: retry once, reminder added to the last user message
            log("\n[NOTE] the model returned an empty reply; retrying once with a reminder.")
            plain = plain[:-1] + [HumanMessage(content=f"{text_of(plain[-1].content)}\n\n{RETRY_REMINDER}")]
            LLM_CALLS += 1
            text = text_of((await llm.ainvoke(plain)).content).strip()
        action = parse_action(text, tool_names)
        if action:                                 # the model asked for a tool: standard tool-call message
            name, args = action
            return {"messages": [AIMessage(content=text, tool_calls=[{"name": name, "args": args, "id": f"call_{LLM_CALLS}"}])]}
        return {"messages": [AIMessage(content=text)]}   # no tool request: this is the final answer

    tool_node = ToolNode(tools)                    # executes the MCP tool calls

    async def run_tools(state: MessagesState, config):
        """Tools node: run the requested MCP tools, and hand the results to the model as plain text.
        (The MCP adapter returns results as a list of content blocks; some providers handle that badly.)"""
        result = await tool_node.ainvoke(state, config)
        for m in result["messages"]:
            if isinstance(m, ToolMessage):
                m.content = text_of(m.content)
        return result

    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model_native if mode == "native" else call_model_react)
    builder.add_node("tools", run_tools)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)  # -> "tools" or END
    builder.add_edge("tools", "agent")
    return builder.compile()


async def ask(graph, question, instructions, use_system_prompt=False):
    """Run one question through the graph, logging every MCP call and response."""
    global LLM_CALLS
    if use_system_prompt:
        messages = [SystemMessage(content=instructions), HumanMessage(content=question)]
    else:
        messages = [HumanMessage(content=f"{instructions}\n\nUser question: {question}")]

    log("=" * 70)
    log(f"USER QUESTION: {question}")
    LLM_CALLS = 0
    final_answer = ""
    last_ai = None

    # stream_mode="updates" yields one dict per executed node: {node_name: output}
    async for update in graph.astream(
        {"messages": messages},
        config={"recursion_limit": MAX_GRAPH_STEPS},
        stream_mode="updates",
    ):
        for node, output in update.items():
            for msg in output["messages"]:
                if node == "agent" and isinstance(msg, AIMessage):
                    last_ai = msg
                    if msg.tool_calls:
                        for call in msg.tool_calls:
                            log(f"\n[MCP CALL]     {call['name']}({json.dumps(call['args'], ensure_ascii=False)})")
                    else:
                        final_answer = text_of(msg.content)
                elif node == "tools" and isinstance(msg, ToolMessage):
                    log(f"\n[MCP RESPONSE] {msg.name}:\n{pretty(text_of(msg.content))}")

    log("\nFINAL ANSWER:")
    if final_answer.strip():
        log(final_answer)
    else:
        log("(the model returned no answer)")
        if last_ai is not None:   # diagnostics to help decide whether to switch models
            log(f"[DEBUG] last model message: finish_reason={last_ai.response_metadata.get('finish_reason')!r}, "
                f"model={last_ai.response_metadata.get('model_name')!r}")
            log("[DEBUG] This model may not handle tool calls reliably; try one whose supported "
                "parameters include 'tools'.")
    log(f"\n(LLM requests used for this question: {LLM_CALLS})")
    return final_answer


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser(description="Weather agent (LangChain + LangGraph + MCP).")
    parser.add_argument("question", nargs="*", help="weather question (omit for interactive mode)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model id (OpenRouter by default)")
    parser.add_argument("--base-url", default=OPENROUTER_URL,
                        help="OpenAI-compatible API base URL (default: OpenRouter)")
    parser.add_argument("--key-env", default="OPENROUTER_API_KEY",
                        help="name of the environment variable that holds the API key")
    parser.add_argument("--mode", choices=["native", "react"], default="native",
                        help="native = provider function calling; react = tool calls written as JSON text "
                             "(for models without reliable tool calling, e.g. Gemma)")
    parser.add_argument("--system-prompt", action="store_true",
                        help="send instructions as a system message instead of inside the user message")
    args = parser.parse_args()

    api_key = os.environ.get(args.key_env)
    if not api_key:
        sys.exit(f"{args.key_env} is not set. In PowerShell: $env:{args.key_env}=\"your-key\"")

    # LLM: OpenRouter exposes an OpenAI-compatible API, so ChatOpenAI works with a custom base_url.
    llm = ChatOpenAI(
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        temperature=0,      # deterministic tool calls and answers
        max_retries=2,      # retry transient errors / rate limits
    )

    # Start both MCP servers and load their tools as LangChain tools.
    client = MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()
    log(f"Model: {args.model} (tool-calling mode: {args.mode})")
    log("MCP tools loaded: " + ", ".join(t.name for t in tools))

    graph = build_graph(llm, tools, args.mode)
    instructions = INSTRUCTIONS if args.mode == "native" else react_instructions(tools)

    questions = [" ".join(args.question)] if args.question else None
    try:
        if questions:
            await ask(graph, questions[0], instructions, args.system_prompt)
        else:
            print("Interactive mode. Type a weather question, or 'quit' to exit.")
            while True:
                q = input("\nYou: ").strip()
                if q.lower() in ("quit", "exit", ""):
                    break
                await ask(graph, q, instructions, args.system_prompt)
    except Exception as err:  # show a hint for the most common failure causes
        log(f"\nERROR: {type(err).__name__}: {err}")
        msg = str(err).lower()
        if "unavailable for free" in msg or "no endpoints" in msg and "tool" not in msg:
            log("Hint: this model id is not available (e.g. its free version was withdrawn). Pick another id on openrouter.ai/models and pass it with --model.")
        elif "tool" in msg:
            log("Hint: this model may not support tool calling. Choose a model whose supported parameters include 'tools'.")
        elif "429" in msg or "rate" in msg:
            log("Hint: rate limit reached (free tiers are limited per day). Wait, or try another model.")
        elif "developer instruction" in msg or "system" in msg:
            log("Hint: this model rejects system messages. Run without --system-prompt.")
        elif "401" in msg or "auth" in msg:
            log("Hint: the API key was rejected. Check the environment variable and that the key is valid.")
        elif "404" in msg:
            log("Hint: model not found. Check the model id spelling on the provider's model page.")
        raise SystemExit(1)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # avoid Windows console encoding errors
    asyncio.run(main())
