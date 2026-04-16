class GroupMonitor:
    """
    A custom node that displays the group of the currently executing node.
    It's a frontend-only display node. The passthrough is to ensure
    it doesn't get pruned from the graph if it's not connected.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                 # This passthrough allows it to be connected to anything,
                 # just to keep it from being pruned by the executor.
                 "passthrough": ("*",)
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "execute"
    CATEGORY = "triXope"

    def execute(self, passthrough=None):
        # This function just passes through any data it receives.
        # Its execution is what triggers the 'executing' message,
        # but our JS logic will mostly ignore this specific node's execution.
        return (passthrough,)