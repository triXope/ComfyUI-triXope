import os

class EnsureDirectoryNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "./my_output_folder", "multiline": False}),
            }
        }

    # Output the confirmed absolute path so it can be passed to other nodes
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("absolute_path",)
    FUNCTION = "ensure_directory"
    CATEGORY = "triXope"
    
    # Setting this to True ensures the node executes even if its output isn't connected to anything
    OUTPUT_NODE = True 

    def ensure_directory(self, directory_path):
        # os.path.abspath resolves relative paths based on the ComfyUI root folder
        abs_path = os.path.abspath(directory_path)
        
        if not os.path.exists(abs_path):
            try:
                # exist_ok=True prevents race condition errors, os.makedirs creates parent folders if needed
                os.makedirs(abs_path, exist_ok=True)
                print(f"[Ensure Directory Node] ✨ Created new directory: {abs_path}")
            except Exception as e:
                print(f"[Ensure Directory Node] ❌ Error creating directory {abs_path}: {e}")
        else:
            print(f"[Ensure Directory Node] ✅ Directory already exists: {abs_path}")
            
        # ComfyUI requires a tuple to be returned
        return (abs_path,)