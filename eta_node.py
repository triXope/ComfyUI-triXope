import server

class RealtimeLoopTracker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "index": ("INT", {"forceInput": True}),
                "total": ("INT", {"forceInput": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("index",)
    FUNCTION = "track_loop"
    CATEGORY = "triXope"
    OUTPUT_NODE = True 

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN") 

    def track_loop(self, index=None, total=None, unique_id=None):
        # If this is the wireless dummy node (no wires connected), quietly do nothing
        if index is None or total is None:
            return (0,)

        node_id = unique_id[0] if isinstance(unique_id, list) else unique_id
        
        server.PromptServer.instance.send_sync("loop_tracker_update", {
            "node_id": node_id,
            "index": index,
            "total": total
        })
        
        return (index,)