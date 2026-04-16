import os

class DirectoryFileSelector:
    """
    A node that takes a directory path and allows selecting a file from it.
    The file list is populated dynamically in the frontend via JavaScript.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"multiline": False, "default": ""}),
                # CHANGE THIS LINE:
                # By changing ([""],) to ("*",), we tell the backend to accept any
                # string value, bypassing the strict list validation.
                "file": ("*",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "get_path"
    CATEGORY = "triXope"

    def get_path(self, directory, file):
        """
        When the workflow is run, this function combines the directory and
        selected file to output the full path.
        """
        # --- Validate Inputs ---
        if not directory or not os.path.isdir(directory):
            print(f"DirectoryFileSelector: The directory path is invalid: {directory}")
            return ("",)

        if not file:
            print(f"DirectoryFileSelector: No file selected in directory: {directory}")
            return ("",)

        # --- Construct and Return Path ---
        full_path = os.path.join(directory, file)
        
        if not os.path.isfile(full_path):
            print(f"DirectoryFileSelector: The selected file does not exist: {full_path}")
            return ("",)
            
        return (full_path,)

# --- Node Registration ---
NODE_CLASS_MAPPINGS = {
    "DirectoryFileSelectorGemini": DirectoryFileSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DirectoryFileSelectorGemini": "triXope Directory File Selector 📂 (Live)"
}