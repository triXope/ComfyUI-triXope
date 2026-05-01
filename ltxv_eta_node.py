class LTXV_ETA_Display:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "real-time_eta": ("ETA_TRACKER", {"forceInput": True, "tooltip": "Connect the real-time_eta output from FilmAuteur_LTXV here."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ()
    FUNCTION = "display_eta"
    CATEGORY = "triXope"
    OUTPUT_NODE = True 

    def display_eta(self, real-time_eta, unique_id=None):
        return ()
