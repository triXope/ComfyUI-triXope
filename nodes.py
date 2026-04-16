import os
import sys
import io
import glob
import shutil
import pathlib
from pathlib import Path
import hashlib
import numpy as np
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchaudio
import random
import math
import folder_paths
import platform
import subprocess
import mimetypes
import json
import struct
import node_helpers
import comfy.model_management
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
from comfy.utils import common_upscale
from comfy.cli_args import args

def is_audio_file(filepath):
    mime_type, _ = mimetypes.guess_type(filepath)
    return mime_type is not None and mime_type.startswith("text/")

class FileCopier:
    CATEGORY = "triXope"
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

def create_vorbis_comment_block(comment_dict, last_block):
    vendor_string = b'ComfyUI'
    vendor_length = len(vendor_string)

    comments = []
    for key, value in comment_dict.items():
        comment = f"{key}={value}".encode('utf-8')
        comments.append(struct.pack('<I', len(comment)) + comment)

    user_comment_list_length = len(comments)
    user_comments = b''.join(comments)

    comment_data = struct.pack('<I', vendor_length) + vendor_string + struct.pack('<I', user_comment_list_length) + user_comments
    if last_block:
        id = b'\x84'
    else:
        id = b'\x04'
    comment_block = id + struct.pack('>I', len(comment_data))[1:] + comment_data

    return comment_block

def insert_or_replace_vorbis_comment(flac_io, comment_dict):
    if len(comment_dict) == 0:
        return flac_io

    flac_io.seek(4)

    blocks = []
    last_block = False

    while not last_block:
        header = flac_io.read(4)
        last_block = (header[0] & 0x80) != 0
        block_type = header[0] & 0x7F
        block_length = struct.unpack('>I', b'\x00' + header[1:])[0]
        block_data = flac_io.read(block_length)

        if block_type == 4 or block_type == 1:
            pass
        else:
            header = bytes([(header[0] & (~0x80))]) + header[1:]
            blocks.append(header + block_data)

    blocks.append(create_vorbis_comment_block(comment_dict, last_block=True))

    new_flac_io = io.BytesIO()
    new_flac_io.write(b'fLaC')
    for block in blocks:
        new_flac_io.write(block)

    new_flac_io.write(flac_io.read())
    return new_flac_io

class AudioFileSaver:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "audio": ("AUDIO", ),
                                "output_path": ("STRING", {"default": folder_paths.get_output_directory()}),
                                "filename_prefix": ("STRING", {"default": "audio"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_audio"
    OUTPUT_NODE = True
    CATEGORY = "triXope"

    def save_audio(self, audio, output_path, filename_prefix="audio", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, os.path.abspath(output_path))
        results = list()

        metadata = {}
        if not args.disable_metadata:
            if prompt is not None:
                metadata["prompt"] = json.dumps(prompt)
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata[x] = json.dumps(extra_pnginfo[x])

        for (batch_number, waveform) in enumerate(audio["waveform"].cpu()):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.flac"

            buff = io.BytesIO()
            torchaudio.save(buff, waveform, audio["sample_rate"], format="FLAC")

            buff = insert_or_replace_vorbis_comment(buff, metadata)

            with open(os.path.join(full_output_folder, file), 'wb') as f:
                f.write(buff.getbuffer())

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "audio": results } }

class LoadText:
    DESCRIPTION = "Load a text file containing screenplay or song lyrics."
    CATEGORY = "triXope"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("OUTPUT",)
    FUNCTION = "load_text"

    @classmethod
    def INPUT_TYPES(s):
        target_dir = os.path.join(folder_paths.get_input_directory(), "_film/input/text")
        exclude_folders = ["clipspace", "folder_to_exclude2"]
        file_list = []

        if os.path.exists(target_dir):
            for root, dirs, files in os.walk(target_dir):
                dirs[:] = [d for d in dirs if d not in exclude_folders]
                for file in files:
                    if file.lower().endswith((".data", ".env", ".info", ".json", ".log", ".text", ".txt", ".yaml", ".yml")):
                        file_list.append(file)

        return {"required": {"text": (sorted(file_list), {"text_upload": False})}}

    def load_text(self, text):
        text_path = os.path.join(folder_paths.get_input_directory(), "_film/input/text", text)
        text_data = ""
        encodings_to_try = ["utf-8", "latin-1", "cp1252", "utf-16", "utf-16-le"]
        for encoding in encodings_to_try:
            try:
                with open(text_path, "r", encoding=encoding) as f:
                    text_data = f.read()
                print(f"File opened successfully with encoding: {encoding}")
                return (text_data,)
            except UnicodeDecodeError:
                print(f"Decoding failed with encoding: {encoding}")
                continue
            except FileNotFoundError:
                print(f"Error: File not found: {text_path}")
                return ("",)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                return ("",)
        print(f"Failed to decode file {text_path} with any of the tested encodings.")
        return ("",)

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
    CATEGORY = "triXope"

    def clean_string(self, text):
        cleaned_text = text.replace(" ", "").replace("\n", "").replace("\r", "")
        return (cleaned_text,)

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
    CATEGORY = "triXope"

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

class ImageCharacterLoader:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        exclude_folders = ["clipspace", "folder_to_exclude2"]
        file_list = []

        for root, dirs, files in os.walk(input_dir + "/_film/input/characters"):
            dirs[:] = [d for d in dirs if d not in exclude_folders]
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                    file_path = os.path.relpath(os.path.join(root, file), start=input_dir).replace("\\", "/")
                    file_list.append(file_path)

        return {"required": {"image": (sorted(file_list), {"image_upload": True})}}

    CATEGORY = "triXope"
    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "load_image"

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        filename_with_ext = os.path.basename(image_path)
        filename, ext = os.path.splitext(filename_with_ext)
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (image, filename)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True

class ImageOutfitLoader:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        exclude_folders = ["clipspace", "folder_to_exclude2"]
        file_list = []

        for root, dirs, files in os.walk(input_dir + "/_film/input/outfits"):
            dirs[:] = [d for d in dirs if d not in exclude_folders]
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                    file_path = os.path.relpath(os.path.join(root, file), start=input_dir).replace("\\", "/")
                    file_list.append(file_path)

        return {"required": {"image": (sorted(file_list), {"image_upload": True})}}

    CATEGORY = "triXope"
    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "load_image"

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        filename_with_ext = os.path.basename(image_path)
        filename, ext = os.path.splitext(filename_with_ext)
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (image, filename)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True

class ImageSceneLoader:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        exclude_folders = ["clipspace", "folder_to_exclude2"]
        file_list = []

        for root, dirs, files in os.walk(input_dir + "/_film/input/scenes"):
            dirs[:] = [d for d in dirs if d not in exclude_folders]
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                    file_path = os.path.relpath(os.path.join(root, file), start=input_dir).replace("\\", "/")
                    file_list.append(file_path)

        return {"required": {"image": (sorted(file_list), {"image_upload": True})}}

    CATEGORY = "triXope"
    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "load_image"

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        filename_with_ext = os.path.basename(image_path)
        filename, ext = os.path.splitext(filename_with_ext)
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (image, filename)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True

class PreviewAnimation:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    methods = {"default": 4, "fastest": 0, "slowest": 6}

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                     "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                     },
                "optional": {
                    "images": ("IMAGE", ),
                },
                }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "triXope"

    def preview(self, fps, images=None):
        filename_prefix = "AnimPreview"
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        results = list()
        pil_images = []

        if images is not None:
            for image in images:
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                pil_images.append(img)

        elif images is not None:
            for image in images:
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                pil_images.append(img)

        else:
            print("PreviewAnimation: No images or masks provided")
            return { "ui": { "images": results, "animated": (None,), "text": "empty" }}

        max_width = 960
        max_height = 540

        scaled_pil_images = []
        for img in pil_images:
            original_size = img.size
            img.thumbnail((max_width, max_height), Image.LANCZOS)
            scaled_pil_images.append(img)
            num_frames = len(pil_images)

        c = len(pil_images)
        for i in range(0, c, num_frames):
            file = f"{filename}_{counter:05}_.webp"
            pil_images[i].save(os.path.join(full_output_folder, file), save_all=True, duration=int(1000.0/fps), append_images=pil_images[i + 1:i + num_frames], lossless=False, quality=80, method=4)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        animated = num_frames != 1
        return { "ui": { "images": results, "animated": (animated,), "text": [f"{num_frames}x{pil_images[0].size[0]}x{pil_images[0].size[1]} (Scaled to {scaled_pil_images[0].size[0]}x{scaled_pil_images[0].size[1]})"] } }

class triXope_VideoPreview:
    @classmethod
    def INPUT_TYPES(s):
        target_dir = os.path.join(folder_paths.get_input_directory(), "_film\\output\\video")
        video_list = []

        if os.path.exists(target_dir):
            for root, dirs, files in os.walk(target_dir):
                for file in files:
                    if file.lower().endswith(('.mp4', '.mov', '.avi', '.webm')):
                        video_list.append(file)

        return {"required": {"video": ("STRING",),}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "triXope"

    def execute(self, video):
        video_path = os.path.join(folder_paths.get_input_directory(), "_film\\output\\video", video)
        video_name = os.path.basename(video_path)
        video_path_name = os.path.dirname(video_path)
        print(f"Selected video: {video_name}")
        print(f"Full video path: {video_path}")
        return {"ui": {"video": [video_name, video_path_name]}}