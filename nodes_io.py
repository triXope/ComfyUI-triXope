"""triXope ComfyUI nodes — File, path and text input/output.

Every node class in this module is registered in the pack's top-level
``__init__.py`` and appears in the ComfyUI node menu under ``triXope/io``.
The module also holds the private helpers those nodes need.
"""

# Imports the moved code relied on from its original module headers.
import shutil

import folder_paths
from aiohttp import web
from server import PromptServer

# --- moved from nodes.py:scan/route/LoadText --------------------------------
ROUTE_PATH = "/triXope/scan_directory"


def clean_path(p):
  """Strips quotes, newlines, and whitespace without altering slashes."""
  if not p:
    return ""
  return str(p).strip(' \t\n\r"\'')


def scan_text_files(directory_input):
  """Scans absolute or relative paths for supported text files."""
  exclude_folders = ["clipspace", "folder_to_exclude2"]
  valid_exts = (
      ".data",
      ".env",
      ".info",
      ".json",
      ".log",
      ".text",
      ".txt",
      ".yaml",
      ".yml",
  )
  file_list = []

  directory_input = clean_path(directory_input)
  if not directory_input:
    directory_input = os.path.join(
        folder_paths.get_input_directory(), "_film/input/text"
    )

  resolved_dir = directory_input
  if not os.path.isabs(resolved_dir):
    maybe_input_path = os.path.join(
        folder_paths.get_input_directory(), resolved_dir
    )
    if os.path.exists(maybe_input_path):
      resolved_dir = maybe_input_path
    else:
      resolved_dir = os.path.abspath(resolved_dir)

  if os.path.exists(resolved_dir) and os.path.isdir(resolved_dir):
    for root, dirs, files in os.walk(resolved_dir):
      dirs[:] = [d for d in dirs if d not in exclude_folders]
      for file in files:
        if file.lower().endswith(valid_exts):
          rel_file = os.path.relpath(os.path.join(root, file), resolved_dir)
          file_list.append(rel_file.replace("\\", "/"))

  return sorted(file_list), resolved_dir


if not any(
    getattr(route, "path", None) == ROUTE_PATH
    for route in PromptServer.instance.routes
):

  @PromptServer.instance.routes.get(ROUTE_PATH)
  async def api_scan_directory(request):
    directory = request.query.get("directory", "")
    files, resolved_path = scan_text_files(directory)
    return web.json_response({"files": files, "resolved_path": resolved_path})


class LoadText:
  DESCRIPTION = (
      "Load a text file containing screenplay or song lyrics from any path."
  )
  CATEGORY = "triXope/io"
  RETURN_TYPES = ("STRING",)
  RETURN_NAMES = ("OUTPUT",)
  FUNCTION = "load_text"

  @classmethod
  def INPUT_TYPES(cls):
    return {
        "required": {
            "directory": (
                "STRING",
                {"default": "_film/input/text", "multiline": False},
            ),
            # Use "*" wildcard to bypass strict backend choice validation
            "file": ("*",),
        }
    }

  def load_text(self, directory, file):
    directory = clean_path(directory)
    file = clean_path(file)

    if not file or file == "None found":
      print("Error: No valid file selected.")
      return ("",)

    files, resolved_dir = scan_text_files(directory)
    text_path = os.path.join(resolved_dir, file)

    encodings_to_try = ["utf-8", "latin-1", "cp1252", "utf-16", "utf-16-le"]
    for encoding in encodings_to_try:
      try:
        with open(text_path, "r", encoding=encoding) as f:
          text_data = f.read()
        print(f"File opened successfully with encoding: {encoding}")
        return (text_data,)
      except UnicodeDecodeError:
        continue
      except FileNotFoundError:
        print(f"Error: File not found at {text_path}")
        return ("",)
      except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return ("",)

    print(f"Failed to decode file {text_path} with any tested encodings.")
    return ("",)

# --- moved from nodes.py:FilenameFromDirectory ------------------------------
class FilenameFromDirectory:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"multiline": False}),
                "file_index": ("INT", {"default": 0, "min": 0}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_filename_by_index"
    CATEGORY = "triXope/io"

    def get_filename_by_index(self, directory_path, file_index):
        import os

        if not os.path.isdir(directory_path):
            raise ValueError(f"The provided path is not a valid directory: {directory_path}")

        files = sorted(
            [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        )

        if not (0 <= file_index < len(files)):
            raise IndexError(f"Index {file_index} is out of range. Directory contains {len(files)} files.")

        filename = files[file_index]
        return (filename,)

# --- moved from nodes.py:FileCopier -----------------------------------------
class FileCopier:
    CATEGORY = "triXope/io"
    NAME = "triXope File Copier"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("STRING", {"forceInput": True}),
                "destination": ("STRING", {}),
                "use_absolute_paths": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "FileCopyFunction"
    OUTPUT_NODE = True

    def FileCopyFunction(self, source, destination, use_absolute_paths):
        source_path = source if use_absolute_paths else os.path.abspath(source)
        dest_path = destination if use_absolute_paths else os.path.abspath(destination)

        source_filename = os.path.basename(source_path)
        dest_path = os.path.join(dest_path, source_filename)

        if not os.path.exists(source_path):
            raise ValueError(f"Source file {source_path} does not exist")

        dest_dir = os.path.dirname(dest_path)
        os.makedirs(dest_dir, exist_ok=True)

        if not os.path.isdir(dest_dir):
            raise ValueError(f"Destination directory {dest_dir} is not a directory")
        if not os.access(dest_dir, os.W_OK):
            raise ValueError(f"No write permission for destination directory {dest_dir}")
        if not os.access(source_path, os.R_OK):
            raise ValueError(f"No read permission for source file {source_path}")

        try:
            shutil.copy2(source_path, dest_path)
            print(f"File copied successfully from {source_path} to {dest_path}")
            return (dest_path,)
        except Exception as e:
            raise ValueError(f"Failed to copy file: {e}")

# --- moved from path_combiner_node.py ---------------------------------------
import os


def sanitize_string(s):
  """Strips surrounding quotes, newlines, and whitespace."""
  if not s:
    return ""
  return str(s).strip(' \t\n\r"\'')


class PathCombinerNode:

  @classmethod
  def INPUT_TYPES(cls):
    inputs = {
        "required": {
            "base_path": ("STRING", {"default": ".", "multiline": False}),
        },
        "optional": {},
    }

    # Dynamically add 10 optional subfolder string inputs
    for i in range(1, 11):
      inputs["optional"][f"subfolder_{i}"] = (
          "STRING",
          {"default": "", "multiline": False},
      )

    return inputs

  # Output 10 strings
  RETURN_TYPES = tuple(["STRING"] * 10)
  RETURN_NAMES = tuple([f"path_{i}" for i in range(1, 11)])
  FUNCTION = "combine_paths"
  CATEGORY = "triXope/io"

  def combine_paths(self, base_path, **kwargs):
    # Clean and resolve the base path to an absolute path
    clean_base = sanitize_string(base_path)
    abs_base_path = os.path.abspath(clean_base)

    results = []
    for i in range(1, 11):
      # Fetch and sanitize subfolder string
      sub_raw = kwargs.get(f"subfolder_{i}", "")
      # Strip quotes, newlines, AND leading/trailing slashes
      sub_clean = str(sub_raw).strip(' \t\n\r"\'\\/')

      # Combine paths safely
      if sub_clean:
        combined = os.path.join(abs_base_path, sub_clean)
      else:
        combined = abs_base_path

      # Normalize path separators and enforce Windows backslashes
      normalized = os.path.normpath(combined).replace("/", "\\")
      results.append(normalized)

    return tuple(results)

# --- moved from ensure_directory.py -----------------------------------------
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
    CATEGORY = "triXope/io"
    
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

# --- moved from text_saver.py -----------------------------------------------
import os

class triXopeTextSaver:
    _file_counters = {}

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "./output", "multiline": False}),
                "filename": ("STRING", {"default": "saved_text", "multiline": False}),
                "text": ("STRING", {"default": "", "multiline": True}),
                "append": ("BOOLEAN", {"default": False}),
                "total_iterations": ("INT", {"default": 1, "min": 1, "max": 99999, "step": 1}),
            },
            "optional": {
                "current_iteration": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "save_text"
    CATEGORY = "triXope/io"

    def save_text(self, path, filename, text, append, total_iterations, current_iteration=0):
        # Create output directory if it does not exist
        if path.strip() and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        # Ensure filename has .txt extension
        if not filename.endswith(".txt"):
            filename += ".txt"

        full_path = os.path.join(path, filename) if path.strip() else filename

        # Determine current iteration count
        if current_iteration > 0:
            run_count = current_iteration
        else:
            # Internal counter tracking if current_iteration input is unlinked
            if not append or not os.path.exists(full_path) or os.path.getsize(full_path) == 0:
                triXopeTextSaver._file_counters[full_path] = 1
            else:
                triXopeTextSaver._file_counters[full_path] = triXopeTextSaver._file_counters.get(full_path, 0) + 1
            run_count = triXopeTextSaver._file_counters[full_path]

        # Determine write mode and format newlines efficiently
        if append and os.path.exists(full_path) and os.path.getsize(full_path) > 0:
            mode = "a"
            with open(full_path, "rb") as f_bin:
                f_bin.seek(-1, os.SEEK_END)
                if f_bin.read(1) != b"\n":
                    text = "\n" + text
        else:
            mode = "a" if append else "w"

        # Write current text chunk to file
        with open(full_path, mode, encoding="utf-8") as f:
            f.write(text)

        # Gate final output: release contents only when total iterations are met
        if run_count >= total_iterations:
            with open(full_path, "r", encoding="utf-8") as f:
                final_contents = f.read()
            triXopeTextSaver._file_counters[full_path] = 0
            return (final_contents,)
        else:
            # Return empty string during loop execution to prevent memory accumulation
            return ("",)

# --- moved from toggle_folder_save.py ---------------------------------------
import os
import json
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

class ToggleFolderSaver:
    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"}),
                "folder_path": ("STRING", {"default": "C:/ComfyUI_Output/", "multiline": False}),
                "filename_prefix": ("STRING", {"default": "custom_save"}),
            },
            # Hidden inputs so the node can embed the workflow metadata into the saved PNG
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    # By returning the image, this acts as a pass-through node
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "triXope/io"

    def save_images(self, images, enabled, folder_path, filename_prefix, prompt=None, extra_pnginfo=None):
        # 1. If disabled, just pass the images through to the next node without saving
        if not enabled:
            return (images,)

        # 2. Ensure the output directory exists
        output_dir = os.path.normpath(folder_path)
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                print(f"[ToggleFolderSaver] Failed to create directory {output_dir}: {e}")
                return (images,)

        # 3. Process and save each image in the batch
        for (batch_number, image) in enumerate(images):
            # Convert the ComfyUI tensor to a PIL Image
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Setup metadata so your ComfyUI workflow is saved inside the PNG file
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            # Find the next available filename by incrementing a counter
            counter = 1
            while True:
                file_name = f"{filename_prefix}_{counter:05d}.png"
                full_path = os.path.join(output_dir, file_name)
                if not os.path.exists(full_path):
                    break
                counter += 1

            # Save the image
            try:
                img.save(full_path, pnginfo=metadata, compress_level=4)
                print(f"[ToggleFolderSaver] Saved image to: {full_path}")
            except Exception as e:
                print(f"[ToggleFolderSaver] Error saving image: {e}")

        # Pass the images through for further use in the workflow
        return (images,)

# --- moved from DirectoryFileSelector.py ------------------------------------
import os
from aiohttp import web
from server import PromptServer

ROUTE_PATH = "/triXope/get_files"


def clean_path(p):
  """Strips quotes, newlines, and whitespace without altering slashes."""
  if not p:
    return ""
  return str(p).strip(' \t\n\r"\'')


if not any(
    getattr(route, "path", None) == ROUTE_PATH
    for route in PromptServer.instance.routes
):

  @PromptServer.instance.routes.get(ROUTE_PATH)
  async def get_files(request):
    directory = clean_path(request.query.get("directory", ""))

    if not directory or not os.path.isdir(directory):
      return web.json_response({"files": []})

    try:
      files = [
          f
          for f in os.listdir(directory)
          if os.path.isfile(os.path.join(directory, f))
      ]
      return web.json_response({"files": sorted(files)})
    except Exception as e:
      print(f"DirectoryFileSelector Error: {e}")
      return web.json_response({"files": []})


class DirectoryFileSelector:
  """A node that takes a directory path and allows selecting a file from it."""

  @classmethod
  def INPUT_TYPES(cls):
    return {
        "required": {
            "directory": ("STRING", {"multiline": False, "default": ""}),
            "file": ("*",),
        },
    }

  RETURN_TYPES = ("STRING",)
  RETURN_NAMES = ("file_path",)
  FUNCTION = "get_path"
  CATEGORY = "triXope/io"

  def get_path(self, directory, file):
    directory = clean_path(directory)
    file = clean_path(file)

    if not directory or not os.path.isdir(directory):
      print(
          f"DirectoryFileSelector: The directory path is invalid: '{directory}'"
      )
      return ("",)

    if not file:
      print(
          f"DirectoryFileSelector: No file selected in directory: '{directory}'"
      )
      return ("",)

    full_path = os.path.join(directory, file)

    if not os.path.isfile(full_path):
      print(
          "DirectoryFileSelector: The selected file does not exist:"
          f" '{full_path}'"
      )
      return ("",)

    return (full_path,)

__all__ = [
    "LoadText",
    "FilenameFromDirectory",
    "FileCopier",
    "PathCombinerNode",
    "EnsureDirectoryNode",
    "triXopeTextSaver",
    "ToggleFolderSaver",
    "DirectoryFileSelector",
]
