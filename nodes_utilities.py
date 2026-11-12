"""triXope ComfyUI nodes — Utilities, switches, routing and monitors.

Every node class in this module is registered in the pack's top-level
``__init__.py`` and appears in the ComfyUI node menu under ``triXope/utilities``.
The module also holds the private helpers those nodes need.
"""

# --- moved from anycast.py --------------------------------------------------
import numbers

try:
    from comfy_execution.graph_utils import ExecutionBlocker
except ImportError:
    try:
        from comfy_execution.graph import ExecutionBlocker
    except ImportError:
        try:
            from execution import ExecutionBlocker
        except ImportError:
            class ExecutionBlocker:
                def __init__(self, value=None):
                    self.value = value


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


ANY_TYPE = AnyType("*")


def try_cast(x, dst_type: str):
    if x is None:
        return None
    result = x
    if dst_type == "STRING":
        result = str(x)
    elif dst_type == "INT":
        result = int(x)
    elif dst_type == "FLOAT" or dst_type == "NUMBER":
        result = float(x)
    elif dst_type == "BOOLEAN":
        if isinstance(x, numbers.Number):
            result = True if x > 0 else False
        elif isinstance(x, str):
            try:
                x = float(x)
                result = True if x > 0 else False
            except:
                result = bool(x)
        else:
            result = bool(x)
    return result


class AnyCast:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "TYPE": (
                    [
                        "*",
                        "STRING",
                        "INT",
                        "FLOAT",
                        "BOOLEAN",
                        "IMAGE",
                        "LATENT",
                        "MASK",
                        "NOISE",
                        "SAMPLER",
                        "SIGMAS",
                        "GUIDER",
                        "MODEL",
                        "CLIP",
                        "VAE",
                        "CONDITIONING",
                        "PYLIST",
                    ],
                    {
                        "tooltip": "ComfyUI type shown on the output socket. STRING, INT, FLOAT, and BOOLEAN also attempt Python conversion."
                    },
                ),
            },
            "optional": {
                "ANY": (ANY_TYPE, {"tooltip": "Value to relabel or convert."}),
            },
        }

    TITLE = "Δ Any Cast"
    DESCRIPTION = "Output any value as the selected ComfyUI type. STRING, INT, FLOAT, and BOOLEAN also perform Python conversion."
    RETURN_TYPES = (ANY_TYPE,)
    OUTPUT_TOOLTIPS = ("The value output as the selected TYPE.",)
    FUNCTION = "run"
    CATEGORY = "triXope/utilities"

    def run(self, TYPE, ANY=None):
        if ANY is None:
            return (ExecutionBlocker(None),)
        
        result = try_cast(ANY, TYPE)
        if result is None:
            return (ExecutionBlocker(None),)
            
        return (result,)

# --- moved from any_switch.py -----------------------------------------------
from .context_utils import is_context_empty
from .utils import FlexibleOptionalInputType, any_type

def is_none(value):
  """Checks if a value is none."""
  if value is not None:
    if isinstance(value, dict) and 'model' in value and 'clip' in value:
      return is_context_empty(value)
  return value is None


class triXopeAnySwitch:
  """Any Switch with graph-aware lazy evaluation and fallback."""

  NAME = "Any Switch"
  CATEGORY = "triXope/utilities"

  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "select": ("INT", {"default": 1, "min": 1, "max": 2, "step": 1}),
        "mode": (["Smart Fallback", "Strict Select"], {"default": "Smart Fallback"}),
      },
      "optional": FlexibleOptionalInputType(
        any_type,
        {
          "any_1": (any_type, {"lazy": True}),
          "any_2": (any_type, {"lazy": True}),
        },
      ),
    }

  RETURN_TYPES = (any_type,)
  RETURN_NAMES = ('*',)
  FUNCTION = "switch"

  def check_lazy_status(self, select=1, mode="Smart Fallback", **kwargs):
    """
    Only requests inputs that are wired in the workflow graph (present in kwargs),
    preventing NodeInputError on unplugged sockets while preserving fallbacks.
    """
    start_idx = max(1, min(2, select))

    if mode == "Strict Select":
      target_key = f"any_{start_idx}"
      if target_key in kwargs:
        return [target_key]
      return []

    # Smart Fallback mode
    for i in range(start_idx, 3):
      key = f"any_{i}"
      # Skip sockets that are completely unplugged in the workflow UI
      if key in kwargs:
        val = kwargs[key]
        # Request evaluation if the connected socket hasn't been evaluated yet
        if val is None:
          return [key]
        # Stop requesting further inputs if a valid (non-None) value is produced
        if not is_none(val):
          return []

    return []

  def switch(self, select=1, mode="Smart Fallback", **kwargs):
    """Chooses the first non-empty item starting from the selected index."""
    inputs = [kwargs.get("any_1"), kwargs.get("any_2")]
    start_index = max(1, min(2, select)) - 1

    any_value = None
    for value in inputs[start_index:]:
      if not is_none(value):
        any_value = value
        break

    return (any_value,)

class triXopeAnySwitchInversed:
  """Any Output Switch with 1 input and 2 strictly locked outputs."""

  NAME = "Any Switch Inversed"
  CATEGORY = "triXope/utilities"

  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "select": ("INT", {"default": 1, "min": 1, "max": 2, "step": 1}),
      },
      "optional": {
        "any_value": (any_type,),
      },
    }

  RETURN_TYPES = (any_type, any_type)
  RETURN_NAMES = ('any_1', 'any_2')
  FUNCTION = "route"

  def route(self, select=1, any_value=None):
    """Strictly routes the value to the exact selected output slot without fallback."""
    outputs = [None, None]
    target_index = max(1, min(2, select)) - 1
    outputs[target_index] = any_value
    return tuple(outputs)

# --- moved from text_switch.py ----------------------------------------------
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
    CATEGORY = "triXope/utilities"

    def check_lazy_status(self, index, **kwargs):
        key = f"text{index}"
        if kwargs.get(key, None) is None:
            return [key]

    def index_switch(self, index, **kwargs):
        key = f"text{index}"
        return (kwargs[key],)

# --- moved from prompt_counter.py -------------------------------------------
from .utils import any_type


class TriXopeCurrentPromptDisplay:
  """A dynamic prompt display node with neon digital formatting showing start, end, and total counts."""

  NAME = "Current Prompt Display"
  CATEGORY = "triXope/utilities"
  OUTPUT_NODE = True

  @classmethod
  def INPUT_TYPES(cls):
    return {
        "required": {},
        "optional": {
            "start": ("INT,FLOAT,STRING", {"forceInput": True}),
            "end": ("INT,FLOAT,STRING", {"forceInput": True}),
            "total": ("INT,FLOAT,STRING", {"forceInput": True}),
            "signal": (any_type, {"lazy": False}),  # Forces execution inside loop
        },
    }

  RETURN_TYPES = (any_type, "STRING")
  RETURN_NAMES = ("signal", "text")
  FUNCTION = "display_prompt"

  @classmethod
  def IS_CHANGED(cls, start=None, end=None, total=None, signal=None):
    return float("nan")

  def display_prompt(self, start=None, end=None, total=None, signal=None):
    display_start = str(start) if start is not None else "?"
    display_end = str(end) if end is not None else "?"
    display_total = str(total) if total is not None else "?"

    raw_text = (
        f"Working on {display_start} to {display_end} of {display_total}."
    )

    neon_html = f"""
    <div style="
        background-color: #050811;
        border: 2px solid #00f3ff;
        border-radius: 10px;
        padding: 12px 18px;
        text-align: center;
        box-shadow: 0 0 15px #00f3ff, inset 0 0 10px #00f3ff;
        font-family: 'Courier New', Courier, monospace;
    ">
      <span style="
          color: #00f3ff;
          font-size: 15px;
          font-weight: bold;
          letter-spacing: 1.2px;
          text-shadow: 0 0 5px #00f3ff;
      ">
        WORKING ON 
        <span style="color: #ff007f; text-shadow: 0 0 8px #ff007f; font-size: 18px;">{display_start}</span> 
        TO 
        <span style="color: #ff007f; text-shadow: 0 0 8px #ff007f; font-size: 18px;">{display_end}</span> 
        OF 
        <span style="color: #ffe600; text-shadow: 0 0 8px #ffe600; font-size: 18px;">{display_total}</span>
      </span>
    </div>
    """

    return {"ui": {"text": [neon_html]}, "result": (signal, raw_text)}

# --- moved from eta_node.py -------------------------------------------------
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
    CATEGORY = "triXope/utilities"
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

# --- moved from ltxv_eta_node.py --------------------------------------------
class LTXV_ETA_Display:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "real_time_eta": ("ETA_TRACKER", {"forceInput": True, "tooltip": "Connect the real-time_eta output from FilmAuteur_LTXV here."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ()
    FUNCTION = "display_eta"
    CATEGORY = "triXope/utilities"
    OUTPUT_NODE = True 

    def display_eta(self, real_time_eta, unique_id=None):
        return ()

# --- moved from GroupMonitor.py ---------------------------------------------
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
    CATEGORY = "triXope/utilities"

    def execute(self, passthrough=None):
        # This function just passes through any data it receives.
        # Its execution is what triggers the 'executing' message,
        # but our JS logic will mostly ignore this specific node's execution.
        return (passthrough,)

# --- moved from final_frame_selector.py -------------------------------------
import torch

class FinalFrameSelector:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_images"
    CATEGORY = "triXope/utilities"

    def process_images(self, images):
        # Handle multiple images
        if images.dim() == 4:  # Multiple images
            final_image = images[-1]

            # Ensure the image is in the correct format
            if final_image.dim() == 3:
                final_image = final_image.unsqueeze(0)

            return (final_image,)

        # Handle single image
        elif images.dim() == 3:
            return (images.unsqueeze(0),)

        else:
            # Unexpected input
            return (None,)

class FirstFrameSelector:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_images"
    CATEGORY = "triXope/utilities"

    def process_images(self, images):
        # Handle multiple images
        if images.dim() == 4:  # Multiple images
            first_image = images[0]

            # Ensure the image is in the correct format
            if first_image.dim() == 3:
                first_image = first_image.unsqueeze(0)

            return (first_image,)

        # Handle single image
        elif images.dim() == 3:
            return (images.unsqueeze(0),)

        else:
            # Unexpected input
            return (None,)

# --- moved from smart_ref_filter.py -----------------------------------------
import torch

class SmartReferenceFilter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "split_prompt": ("STRING", {"forceInput": True}),
                "ref_image_batch": ("IMAGE",),
                "character_names": ("STRING", {
                    "default": "JOHN DOE, JANE DOE, ANY", 
                    "multiline": True
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("filtered_ref_image", "has_character")
    FUNCTION = "filter_reference"
    CATEGORY = "triXope/utilities"

    def filter_reference(self, split_prompt, ref_image_batch, character_names):
        # Parse comma-separated names into a list
        names_list = [n.strip() for n in character_names.split(",") if n.strip()]
        prompt_lower = split_prompt.lower()
        batch_size = ref_image_batch.shape[0]
        
        matched_indices = []

        # Iterate over batch indices mapped to provided names
        for i in range(min(batch_size, len(names_list))):
            target_name = names_list[i].lower()
            
            # Match if wildcard "ANY" or if target name is present in prompt
            if target_name == "any" or target_name in prompt_lower:
                matched_indices.append(i)
        
        if matched_indices:
            # Filter tensor batch to only keep matched indices
            filtered_batch = ref_image_batch[matched_indices]
            print(f"[SmartReferenceFilter] Matched index/indices {matched_indices}. Passing filtered batch.")
            return (filtered_batch, True)
        else:
            print(f"[SmartReferenceFilter] No character matches or wildcards found. Supplying zero-filled reference batch.")
            blank_batch = torch.zeros(ref_image_batch.shape, dtype=torch.float32)
            return (blank_batch, False)

class SmartConditioningSelector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_prompt": ("STRING", {"forceInput": True}),
                "pos_with_ref": ("CONDITIONING",),
                "latent_with_ref": ("LATENT",),
                "pos_no_ref": ("CONDITIONING",),
                "latent_no_ref": ("LATENT",),
                "character_names": ("STRING", {
                    "default": "BLAKE, SYLVIA, CARY, KARAN",
                    "multiline": True
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "select_branch"
    CATEGORY = "triXope/utilities"

    def select_branch(self, text_prompt, pos_with_ref, latent_with_ref, pos_no_ref, latent_no_ref, character_names):
        names = [n.strip().lower() for n in character_names.split(",") if n.strip()]
        prompt_lower = text_prompt.lower()
        
        has_character = any(name in prompt_lower for name in names)
        
        if has_character:
            print("[SmartConditioningSelector] Character detected -> Routing Reference Branch.")
            return (pos_with_ref, latent_with_ref)
        else:
            print("[SmartConditioningSelector] No character detected -> Routing Clean T2V Branch.")
            return (pos_no_ref, latent_no_ref)

# --- moved from nodes.py:StringCleaner --------------------------------------
class StringCleaner:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "clean_string"
    CATEGORY = "triXope/utilities"

    def clean_string(self, text):
        cleaned_text = text.replace(" ", "").replace("\n", "").replace("\r", "")
        return (cleaned_text,)

# --- moved from __init__.py:WorkflowGroupHighlighter ------------------------
class WorkflowGroupHighlighter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = "triXope/utilities"
    OUTPUT_NODE = True

    def execute(self):
        return ()

__all__ = [
    "AnyCast",
    "triXopeAnySwitch",
    "triXopeAnySwitchInversed",
    "text_switch",
    "TriXopeCurrentPromptDisplay",
    "RealtimeLoopTracker",
    "LTXV_ETA_Display",
    "GroupMonitor",
    "FinalFrameSelector",
    "FirstFrameSelector",
    "SmartReferenceFilter",
    "SmartConditioningSelector",
    "StringCleaner",
    "WorkflowGroupHighlighter",
]
