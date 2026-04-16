MAX_FLOW_NUM = 100

comfy_ui_revision = None
def get_comfyui_revision():
    try:
        import git
        import os
        import folder_paths
        repo = git.Repo(os.path.dirname(folder_paths.__file__))
        comfy_ui_revision = len(list(repo.iter_commits('HEAD')))
    except:
        comfy_ui_revision = "Unknown"
    return comfy_ui_revision

def compare_revision(num):
    global comfy_ui_revision
    if not comfy_ui_revision:
        comfy_ui_revision = get_comfyui_revision()
    return True if comfy_ui_revision == 'Unknown' or int(comfy_ui_revision) >= num else False

lazy_options = {"lazy": True} if compare_revision(2543) else {}

class text_switch:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": 9, "step": 1}),
            },
            "optional": {}
        }
        for i in range(MAX_FLOW_NUM):
            inputs["optional"][f"text{i}"] = ("STRING", {**lazy_options, "forceInput": True})
        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "index_switch"
    CATEGORY = "triXope"

    def check_lazy_status(self, index, **kwargs):
        key = f"text{index}"
        if kwargs.get(key, None) is None:
            return [key]

    def index_switch(self, index, **kwargs):
        key = f"text{index}"
        return (kwargs[key],)