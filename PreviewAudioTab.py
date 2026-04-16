import os
import torch
import torchaudio
import folder_paths
import uuid

class PreviewAudioTabNode:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_TabPreview"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", ),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "preview_audio"
    OUTPUT_NODE = True
    CATEGORY = "trIxope"

    def preview_audio(self, audio):
        # Check if the input is the expected dictionary format
        if not isinstance(audio, dict) or 'waveform' not in audio or 'sample_rate' not in audio:
            print("Warning: Invalid audio input format for PreviewAudioTabNode.")
            return {"ui": {"audio": []}}

        waveform = audio['waveform'] # Should be [1, samples]
        sample_rate = audio['sample_rate']

        # Ensure waveform is a tensor and has the expected 2D shape
        if not isinstance(waveform, torch.Tensor):
            print("Warning: Waveform is not a tensor in PreviewAudioTabNode.")
            return {"ui": {"audio": []}}
            
        if waveform.ndim != 2 or waveform.shape[0] != 1:
            print(f"Warning: Expected 2D waveform [1, samples], got shape {waveform.shape}. Attempting to reshape.")
            # Try to reshape common errors
            if waveform.ndim == 1: # If it's 1D, add the channel dimension
                 waveform = waveform.unsqueeze(0)
            elif waveform.ndim == 2 and waveform.shape[0] > 1: # If it's stereo/multi-channel, take the first channel
                 waveform = waveform[0, :].unsqueeze(0)
            elif waveform.ndim > 2: # Too many dimensions
                 print("Error: Waveform has too many dimensions.")
                 return {"ui": {"audio": []}}
            # Check again after potential reshape
            if waveform.ndim != 2 or waveform.shape[0] != 1:
                 print("Error: Could not reshape waveform to [1, samples].")
                 return {"ui": {"audio": []}}


        # --- Prepare filename ---
        # Use a unique ID to prevent filename collisions in temp folder
        filename_prefix = f"{self.prefix_append}_{uuid.uuid4()}"
        full_output_folder, filename, counter, subfolder, filename_prefix_out = folder_paths.get_save_image_path(filename_prefix, self.output_dir, 1, 1) # Simplified call for temp files

        # Use WAV for broad compatibility, though FLAC is smaller
        file = f"{filename_prefix_out}_{counter:05}_.wav"
        file_path = os.path.join(full_output_folder, file)

        try:
            # --- IMPORTANT: Squeeze to 1D *only* for saving the preview ---
            # torchaudio.save expects [channels, samples] or [samples]
            # Since we ensured it's [1, samples], squeezing is safe here.
            waveform_to_save = waveform.squeeze(0)

            # Save the audio file
            torchaudio.save(file_path, waveform_to_save, sample_rate)

            # --- Result for ComfyUI UI ---
            results = [{
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            }]
            
            # Debug print
            print(f"PreviewAudioTabNode: Saved preview {file} (Shape: {waveform_to_save.shape}, SR: {sample_rate})")


        except Exception as e:
            print(f"Error saving audio preview in PreviewAudioTabNode: {e}")
            results = []


        return {"ui": {"audio": results}}