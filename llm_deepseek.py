import llm
import httpx

@llm.hookimpl
def register_models(register):
    register(DeepSeek("deepseek-chat"))
    register(DeepSeek("deepseek-coder"))

class DeepSeek(llm.Model):
    
    def __init__(self, model_id):
        self.model_id = model_id
    
    def build_messages(self, prompt, conversation):
        messages = []
        if not conversation:
            if prompt.system:
                messages.append({"role": "system", "content": prompt.system})
            messages.append({"role": "user", "content": prompt.prompt})
            return messages
        current_system = None
        for prev_response in conversation.responses:
            if (
                prev_response.prompt.system
                and prev_response.prompt.system != current_system
            ):
                messages.append(
                    {"role": "system", "content": prev_response.prompt.system}
                )
                current_system = prev_response.prompt.system
            messages.append({"role": "user", "content": prev_response.prompt.prompt})
            messages.append({"role": "assistant", "content": prev_response.text()})
        if prompt.system and prompt.system != current_system:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": prompt.prompt})
        return messages
    
    def execute(self, prompt: str, stream, response, conversation):
        key = llm.get_key("", "deepseek", "LLM_DEEPSEEK_KEY") or getattr(
            self, "key", None
        )
        messages = self.build_messages(prompt, conversation)
        response._prompt_json = {"messages": messages}
        body = {
            "model": self.model_id,
            "messages": messages,
        }
    
        with httpx.Client() as client:
            api_response = client.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {key}",
                },
                json=body,
                timeout=None,
            )
            api_response.raise_for_status()
            yield api_response.json()["choices"][0]["message"]["content"]
            response.response_json = api_response.json()