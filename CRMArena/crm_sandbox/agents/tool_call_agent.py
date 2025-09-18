import json
from typing import Dict, List, Any
import time
from crm_sandbox.openrouter_client import chat_completion
from tenacity import retry, stop_after_attempt, wait_random_exponential
from crm_sandbox.agents.prompts import SCHEMA_STRING, SYSTEM_METADATA, NATIVE_FC_PROMPT, FC_RULE_STRING, FC_FLEX_PROMPT
from crm_sandbox.agents.utils import OPENROUTER_MODELS_MAP


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(10))
def chat_completion_request(
    messages,
    model,
    tools=None,
    temperature: float = 0.0,
    top_p=1.0,
    max_tokens=3500
):
    return chat_completion(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        tools=tools,
        extra_body={"usage": {"include": True}},
    )
    
class ToolCallAgent:
    def __init__(
        self, tools, schema_obj, model: str = "google/gemini-2.5-flash-lite", max_turns: int = 20, eval_mode="default", strategy="tool_call", provider="openrouter"
    ):
        schema = self._build_schema(schema_obj)
        self.tools = tools
        
        if strategy == "tool_call":
            self.sys_prompt = NATIVE_FC_PROMPT.format(system="Salesforce instance")
        else:
            self.sys_prompt = FC_FLEX_PROMPT.format(system_description=schema, system="Salesforce instance")
            
        self.model = model
        self.eval_mode = eval_mode
        self.max_turns = max_turns
        self.usage = {"cost": [], "completion_tokens": [], "prompt_tokens": [], "total_tokens": []}
        if provider != "openrouter":
            raise ValueError("Only the openrouter provider is supported")
        self.provider = provider
        if self.model in OPENROUTER_MODELS_MAP:
            self.model = OPENROUTER_MODELS_MAP[self.model]["name"]
            

    def _build_schema(self, schema_obj):
        object_description = dict()
        for item in schema_obj:
            object_description[item["object"]] = "\n".join([f"  - {k}: {v}" for k,v in item["fields"].items()])
            
        template = SCHEMA_STRING.format(
            object_names=", ".join(object_description.keys()),
            object_fields="\n".join(
                [f"{obj}\n{fields}" for obj, fields in object_description.items()]
            )
        )
        return template
    
    def reset(self, args):
        if args["metadata"]["required"]:
            self.sys_prompt += SYSTEM_METADATA.format(system_metadata=args["metadata"]["required"], system="Salesforce instance") # add task/query-specific metadata here
        if self.eval_mode == "aided" and "optional" in args["metadata"]:
            self.sys_prompt += "\n" + args["metadata"]["optional"]
        self.messages = [{"role": "system", "content": self.sys_prompt.strip()}, {"role": "user", "content": args["query"].strip()}]
        self.usage = {"cost": [], "completion_tokens": [], "prompt_tokens": [], "total_tokens": []}
        
    def act(self, env, index=None, temperature=0.0):
        query, metadata = env.reset(task_index=index)
        self.reset({"query": query, "metadata": metadata})
        self.info = {}
        self.info["observation_sizes"] = []
        done = False
        reward = 0
        info = {}

        for turn_id in range(self.max_turns):
            time.sleep(3)
            info = {}
            res = chat_completion_request(
                messages=self.messages,
                model=self.model,
                temperature=0.0,
                top_p=1.0,
                max_tokens=3500,
                tools=self.tools
            )
            message = res.choices[0].message.model_dump()

            usage = res.usage
            usage_dict = {}
            if usage is not None:
                if hasattr(usage, "model_dump"):
                    usage_dict = usage.model_dump()
                elif isinstance(usage, dict):
                    usage_dict = usage
            for key in self.usage.keys():
                if key != "cost":
                    self.usage[key].append(usage_dict.get(key, 0))
            cost = usage_dict.get("cost") or usage_dict.get("total_cost")
            if cost is None:
                hidden = getattr(res, "_hidden_params", {}) or {}
                cost = hidden.get("response_cost") or hidden.get("cost")
            self.usage["cost"].append(cost)

            print("message", message, flush=True)
            action = self.message_action_parser(message)
            print("#", turn_id, "Agent action:", action, flush=True)

            if action is None:
                self.info["end_reason"] = {
                    "source": "agent",
                    "message": "Invalid action",
                    "content": message["content"].strip()
                }
                info["end_reason"] = self.info["end_reason"]
                self.messages.append(message)
                self.messages.append({"role": "user", "content": FC_RULE_STRING})
                continue

            if "tool_calls" in message and message["tool_calls"]:
                message["tool_calls"] = message["tool_calls"][:1]
            self.messages.append(message)

            obs, reward, done, info = env.step(action)
            if "observation_size" in info:
                self.info["observation_sizes"].append(info["observation_size"])
            if "end_reason" in info:
                self.info["end_reason"] = info["end_reason"]
            if done:
                break

            tool_call_id = None
            if "tool_calls" in message and message["tool_calls"]:
                tool_call_id = (message["tool_calls"][0].get("id") or "").strip()
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id or action["name"],
                    "name": action["name"],
                    "content": obs
                }
            )

        if not done:
            if "end_reason" not in info:
                self.info["end_reason"] = {
                    "source": "agent",
                    "message": "Max turns reached",
                    "content": message["content"].strip()
                }
        self.info["usage"] = self.usage
        self.info["total_cost"] = sum(cost for cost in self.usage["cost"] if cost is not None)
        self.info["num_turns"] = turn_id + 1
        return reward

    def get_messages(self) -> List[Dict[str, str]]:
        return self.messages

    def message_action_parser(self, message: Dict[str, Any]) -> Dict[str, Any]:
        if "tool_calls" in message and message["tool_calls"] and message["tool_calls"][0]["function"] is not None:
            tool_call = message["tool_calls"][0]
            try:
                return {
                    "name": tool_call["function"]["name"].strip(),
                    "arguments": json.loads(tool_call["function"]["arguments"].strip()),
                }
            except json.JSONDecodeError:
                return None
        return None
