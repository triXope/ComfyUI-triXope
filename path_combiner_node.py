import os

class PathCombinerNode:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "base_path": ("STRING", {"default": ".", "multiline": False}),
            },
            "optional": {}
        }
        
        # Dynamically add 10 optional subfolder string inputs
        for i in range(1, 11):
            inputs["optional"][f"subfolder_{i}"] = ("STRING", {"default": "", "multiline": False})
            
        return inputs

    # Output 10 strings
    RETURN_TYPES = tuple(["STRING"] * 10)
    RETURN_NAMES = tuple([f"path_{i}" for i in range(1, 11)])
    FUNCTION = "combine_paths"
    CATEGORY = "triXope"

    def combine_paths(self, base_path, **kwargs):
        # Resolve the base path to a full absolute path
        abs_base_path = os.path.abspath(base_path)
        
        results = []
        for i in range(1, 11):
            # Fetch the subfolder string, default to empty if not provided
            sub = kwargs.get(f"subfolder_{i}", "").strip()
            
            # Combine paths if a subfolder exists, otherwise use just the base path
            if sub:
                combined = os.path.join(abs_base_path, sub)
            else:
                combined = abs_base_path
                
            # Enforce the requested backslash delimiter for all slashes
            combined = combined.replace("/", "\\")
            results.append(combined)
                
        return tuple(results)