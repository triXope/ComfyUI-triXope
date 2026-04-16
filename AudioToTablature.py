import os
import torch
import numpy as np
import folder_paths
from PIL import Image, ImageDraw, ImageFont
import io

# --- Library Imports ---
try:
    import librosa
    # We no longer need music21
except ImportError:
    print("-----------------------------------------------------------------")
    print("ERROR: AudioToTablature node failed to load required libraries.")
    print("Please install 'librosa' into ComfyUI's Python environment.")
    print("If you use a portable install, run this in the 'python_embeded' folder:")
    print("python.exe -m pip install librosa")
    print("Also ensure FFmpeg is installed and accessible in your system's PATH.")
    print("-----------------------------------------------------------------")


# --- Helper Function: Pitch-to-Fret Logic ---
def get_simplest_fret(midi_note, tuning=[64, 59, 55, 50, 45, 40]):
    """
    Finds the simplest fret/string combination for a given MIDI note.
    'tuning' is the MIDI notes of the open strings, from 1st (high E) to 6th (low E).
    [64, 59, 55, 50, 45, 40] = [E4, B3, G3, D3, A2, E2]
    """
    best_fret = 99
    best_string = -1
    for string_index, open_note in enumerate(tuning):
        string_num = string_index + 1
        fret = midi_note - open_note
        if 0 <= fret < best_fret:
            best_fret = fret
            best_string = string_num
    if best_string == -1: return None
    return (best_string, best_fret)


# --- Helper Function: Text-to-Image Conversion ---
def text_to_pil_image(text_content, font_size=16, padding=20):
    """ Converts a block of text into a PIL Image. """
    try:
        font = ImageFont.truetype("cour.ttf", font_size) # Courier New
    except IOError:
        try: font = ImageFont.truetype("DejaVuSansMono.ttf", font_size) # Linux fallback
        except IOError:
            print("Warning: Monospaced font not found. Using default PIL font.")
            font = ImageFont.load_default()

    dummy_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    if text_content is None: text_content = "Error: No text generated."
    lines = text_content.split('\n')
    line_bboxes = [dummy_draw.textbbox((0, 0), line, font=font) for line in lines]
    line_widths = [bbox[2] - bbox[0] for bbox in line_bboxes]
    line_heights = [bbox[3] - bbox[1] for bbox in line_bboxes]

    if not line_widths or not line_heights: # Handle empty text
        blank_img = Image.new('RGB', (padding*2 + 50, padding*2 + font_size), color='white')
        blank_output = np.array(blank_img).astype(np.float32) / 255.0
        return torch.from_numpy(blank_output).unsqueeze(0)

    text_width = max(line_widths)
    total_height = sum(line_heights) + (len(lines) -1) * (font_size * 0.2)
    img_width = int(text_width + (padding * 2))
    img_height = int(total_height + (padding * 2))
    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)
    y_text = padding
    for i, line in enumerate(lines):
        draw.text((padding, y_text), line, font=font, fill='black')
        y_text += line_heights[i] + (font_size * 0.2)
    output_image = np.array(img).astype(np.float32) / 255.0
    output_image = torch.from_numpy(output_image).unsqueeze(0)
    return output_image

# --- Helper Function: MIDI Notes-to-Audio Synthesis (Outputs 2D Audio) ---
def generate_audio_preview(notes_list, note_duration_sec=0.25, sample_rate=44100):
    """ Generates audio preview, returns 2D tensor [1, samples]. """
    if not notes_list: return {"waveform": torch.zeros(1, sample_rate // 4).float(), "sample_rate": sample_rate}
    all_waveforms = []
    samples_per_note = int(note_duration_sec * sample_rate)
    if samples_per_note <= 0: samples_per_note = int(0.05 * sample_rate)
    t = np.linspace(0., note_duration_sec, samples_per_note, endpoint=False)
    fade_out = np.linspace(1, 0, samples_per_note)
    for midi_note in notes_list:
        freq = librosa.midi_to_hz(midi_note)
        waveform = np.sin(2 * np.pi * freq * t) * fade_out * 0.5
        all_waveforms.append(waveform)
    if not all_waveforms: return {"waveform": torch.zeros(1, sample_rate // 4).float(), "sample_rate": sample_rate}
    try: final_waveform_np = np.concatenate(all_waveforms).astype(np.float32)
    except ValueError: return {"waveform": torch.zeros(1, sample_rate // 4).float(), "sample_rate": sample_rate}
    final_waveform_tensor = torch.from_numpy(final_waveform_np).unsqueeze(0)
    return {"waveform": final_waveform_tensor, "sample_rate": sample_rate}


# --- The ComfyUI Custom Node ---
class AudioToTablature:
    """
    Analyzes audio waveform, detects pitches, generates guitar tablature & audio preview.
    Accepts AUDIO input (1D, 2D, or 3D). Returns silent audio on failure.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "time_step": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
                "confidence_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "notes_per_measure": ("INT", {"default": 16, "min": 1, "max": 16, "step": 1}),
                "measures_per_line": ("INT", {"default": 4, "min": 1, "max": 10, "step": 1}),
                "note_duration": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO")
    RETURN_NAMES = ("tablature_image", "tablature_text", "audio_preview")
    FUNCTION = "generate_tab"
    CATEGORY = "triXope"

    def generate_tab(self, audio, time_step, confidence_threshold, notes_per_measure, measures_per_line, note_duration):
        default_sr = 44100
        silent_audio = {"waveform": torch.zeros(1, default_sr // 4).float(), "sample_rate": default_sr}

        # --- 1. Extract and Validate Audio Data ---
        if not isinstance(audio, dict) or 'waveform' not in audio or 'sample_rate' not in audio:
            error_msg = "Error: Invalid AUDIO input format."
            return (text_to_pil_image(error_msg), error_msg, silent_audio)

        waveform_tensor = audio['waveform']
        sr = audio['sample_rate']
        silent_audio = {"waveform": torch.zeros(1, sr // 4).float(), "sample_rate": sr} # Update silence SR

        # --- MODIFIED: Handle 3D input [batch, channels, samples] ---
        if waveform_tensor.ndim == 3:
            # Assume batch size is 1 and remove it
            print(f"Info: Input audio tensor had shape {waveform_tensor.shape}. Squeezing batch dimension.")
            waveform_tensor = waveform_tensor.squeeze(0)
        # Now waveform_tensor should be 2D [channels, samples] or 1D [samples]
        # -------------------------------------------------------------

        # --- Convert to NumPy & Ensure Mono/1D ---
        try:
            y_np = waveform_tensor.cpu().numpy().astype(np.float32)
        except Exception as e:
            error_msg = f"Error converting audio tensor to NumPy: {e}"
            return (text_to_pil_image(error_msg), error_msg, silent_audio)

        # Ensure y is 1D (mono)
        if y_np.ndim == 0:
            error_msg = f"Error: Input audio waveform is a 0-D array (single value)."
            return (text_to_pil_image(error_msg), error_msg, silent_audio)
        elif y_np.ndim == 1:
            # Already mono
            print("Info: Input audio is mono.")
            y = y_np
        elif y_np.ndim == 2:
            # Average channels if shape is [Channels, Samples]
             print(f"Info: Input audio tensor had shape {y_np.shape} [Channels, Samples?]. Averaging channels to mono.")
             y = np.mean(y_np, axis=0) # Average across axis 0 (channels)
        else: # Should not happen after squeeze, but good to check
            error_msg = f"Error: Input audio waveform has unexpected shape {y_np.shape} after processing."
            return (text_to_pil_image(error_msg), error_msg, silent_audio)

        # Final check: Ensure y is definitely 1D
        if y.ndim != 1:
            error_msg = f"Error: Failed to convert audio to 1D NumPy array after averaging. Final shape is {y.shape}."
            return (text_to_pil_image(error_msg), error_msg, silent_audio)

        if y.size == 0:
            error_msg = "Error: Input audio waveform is empty after processing."
            return (text_to_pil_image(error_msg), error_msg, silent_audio)

        # --- Ensure C-Contiguous Memory Layout ---
        if not y.flags['C_CONTIGUOUS']:
            y = np.ascontiguousarray(y)
            print("Info: Converted audio array to C-contiguous.")


        # --- 2. Analyze Audio with Librosa ---
        try:
            print(f"Detecting pitch (f0) using librosa.pyin on audio of shape {y.shape}...")
            f0, voiced_flag, voiced_prob = librosa.pyin(
                y, # Pass the guaranteed 1D, C-contiguous array
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr,
                frame_length=2048
            )

            timestamps = librosa.times_like(f0, sr=sr)
            notes = []

            for i in range(len(timestamps)):
                if voiced_prob[i] > confidence_threshold and np.isfinite(f0[i]) and f0[i] > 0:
                    timestamp = timestamps[i]
                    midi_note = int(round(librosa.hz_to_midi(f0[i])))
                    notes.append((timestamp, midi_note))

        except ValueError as e: # Catch the specific "ambiguous truth value" error
             if "ambiguous" in str(e):
                 error_msg = f"Error: Pitch detection failed (ambiguous array value). Details: {e}"
             else:
                 error_msg = f"Error: Could not detect pitch. {e}. "
             print(error_msg)
             return (text_to_pil_image(error_msg), error_msg, silent_audio)

        except Exception as e: # Catch other potential errors
            print(f"Error during pitch detection: {e}")
            error_msg = f"Error: Could not detect pitch. {e}. "
            return (text_to_pil_image(error_msg), error_msg, silent_audio)


        if not notes:
            error_msg = "No notes with high confidence found in the audio."
            return (text_to_pil_image(error_msg), error_msg, silent_audio)

        # --- 3. De-duplicate and group notes ---
        grouped_notes = []
        last_note = -1
        last_time = -1
        for timestamp, midi_note in notes:
            if midi_note != 0:
                if midi_note != last_note:
                    grouped_notes.append(midi_note)
                    last_note = midi_note
                    last_time = timestamp
                elif (timestamp - last_time) > time_step:
                    grouped_notes.append(midi_note)
                    last_time = timestamp

        # --- 4. Manually Build ASCII Tablature ---
        tab_lines = ['e|', 'B|', 'G|', 'D|', 'A|', 'E|']
        notes_in_current_measure = 0
        measures_in_current_line = 0
        current_line_offset = 0
        for midi_note in grouped_notes:
            if notes_in_current_measure >= notes_per_measure:
                for i in range(6): tab_lines[current_line_offset + i] += '|'
                notes_in_current_measure = 0
                measures_in_current_line += 1
                if measures_in_current_line >= measures_per_line:
                    tab_lines.extend(['', 'e|', 'B|', 'G|', 'D|', 'A|', 'E|'])
                    current_line_offset = len(tab_lines) - 6
                    measures_in_current_line = 0
            pos = get_simplest_fret(midi_note)
            note_column = ['-', '-', '-', '-', '-', '-']
            if pos is not None:
                (string_num, fret_num) = pos
                string_index = string_num - 1
                note_column[string_index] = str(fret_num)
            max_width = max(len(s) for s in note_column) if note_column else 1
            for i in range(6):
                s = note_column[i]
                padded_s = s.center(max_width, '-')
                if current_line_offset + i < len(tab_lines):
                     tab_lines[current_line_offset + i] += padded_s + "-"
            notes_in_current_measure += 1
        for i in range(6):
            if current_line_offset + i < len(tab_lines): tab_lines[current_line_offset + i] += '|'
        tab_text = "\n".join(tab_lines)

        # --- 5. Generate Text and Image Outputs ---
        if not tab_text.strip(): tab_text = "No tablature could be generated."
        tab_image = text_to_pil_image(tab_text)

        # --- 6. Generate Audio Preview (Outputs 2D Audio) ---
        audio_output = generate_audio_preview(grouped_notes, note_duration, sample_rate=sr)

        return (tab_image, tab_text, audio_output)