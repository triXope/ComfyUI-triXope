class TriXopePromptSplitSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "first part | second part | third part"}),
                "index": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "select_prompt"
    CATEGORY = "triXope"

    def select_prompt(self, text: str, index: int):
        parts = [p.strip() for p in text.split('|')]
        parts = [p for p in parts if p]
        if not parts:
            return ("",)
        safe_index = index % len(parts)
        selected_part = parts[safe_index]
        return (selected_part,)

print("### Loading: triXope Prompt Split & Select ###")