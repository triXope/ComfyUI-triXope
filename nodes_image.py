"""triXope ComfyUI nodes — Image processing, colour and preview.

Every node class in this module is registered in the pack's top-level
``__init__.py`` and appears in the ComfyUI node menu under ``triXope/image``.
The module also holds the private helpers those nodes need.
"""

# --- moved from ColorFX.py --------------------------------------------------
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageChops, ImageDraw
import cv2
import math
import random
import os
import folder_paths

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    from scipy.interpolate import RegularGridInterpolator
    SCIPY_AVAILABLE = True
    print("INFO: Scipy found. LUT application will be optimized.")
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: Scipy not found. LUT application will use a slower pixel-by-pixel method if enabled.")

try:
    dir_luts = os.path.join(folder_paths.models_dir, "luts")
    os.makedirs(dir_luts, exist_ok=True)
    if "luts" not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths["luts"] = ([dir_luts], {".cube"})
    else:
        folder_paths.folder_names_and_paths["luts"][1].add(".cube")
        if dir_luts not in folder_paths.folder_names_and_paths["luts"][0]:
             folder_paths.folder_names_and_paths["luts"][0].append(dir_luts)
except Exception as e:
    print(f"Error setting up LUTs directory: {e}. LUT functionality may be limited.")


def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class ColorFX:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        lut_files = ["None"]
        try:
            raw_files = folder_paths.get_filename_list("luts")
            if raw_files:
                 lut_files.extend([f for f in raw_files if f.lower().endswith('.cube')])
        except Exception as e:
            print(f"Warning: Could not load LUT files for dropdown: {e}")

        return {
            "required": {
                # --- MASTER BYPASS & STAGE TOGGLES ---
                "images": ("IMAGE",),
                "enable": ("BOOLEAN", {"default": True}),
                "enable_color_correction": ("BOOLEAN", {"default": True}),
                "enable_lut_processing": ("BOOLEAN", {"default": True}),
                "enable_enhancements": ("BOOLEAN", {"default": True}),
                "enable_blur_effects": ("BOOLEAN", {"default": True}),
                "enable_stylistic_effects": ("BOOLEAN", {"default": True}),

                # --- GROUP: Color Correction ---
                "grp_color_correction": (["Color Correction"], {}),
                "hdr_intensity": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "shadow_intensity": ("FLOAT", {"default": 0.10, "min": 0.00, "max": 2.00, "step": 0.01}),
                "highlight_intensity": ("FLOAT", {"default": 0.20, "min": 0.00, "max": 2.00, "step": 0.01}),
                "gamma": ("FLOAT", {"default": 1.00, "min": 0.10, "max": 5.00, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 0.90, "min": 0.00, "max": 3.00, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 0.75, "min": 0.00, "max": 3.00, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 0.90, "min": 0.00, "max": 3.00, "step": 0.01}),
                "enhance_color": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 3.00, "step": 0.01}),

                # --- GROUP: LUT Processing ---
                "grp_lut": (["LUT Processing"], {}),
                "lut_name": (lut_files, {"default": "None"}),
                "lut_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "lut_log_process": ("BOOLEAN", {"default": True}),

                # --- GROUP: Enhancements ---
                "grp_enhancements": (["Enhancements"], {}),
                "sharpness": ("FLOAT", {"default": 3.0, "min": -2.0, "max": 5.0, "step": 0.1}),
                "edge_enhance_strength": ("FLOAT", {"default": 0.00, "min": 0.00, "max": 1.00, "step": 0.01}),
                "detail_enhance_strength": ("FLOAT", {"default": 0.20, "min": 0.00, "max": 1.00, "step": 0.01}),

                # --- GROUP: Blur Effects ---
                "grp_blur": (["Blur Effects"], {}),
                "blur_radius": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "gaussian_blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 64.0, "step": 0.1}),
                "radial_blur_strength": ("FLOAT", {"default": 32.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "radial_blur_center_x": ("FLOAT", {"default": 0.50, "min": 0.00, "max": 1.00, "step": 0.01}),
                "radial_blur_center_y": ("FLOAT", {"default": 0.50, "min": 0.00, "max": 1.00, "step": 0.01}),
                "radial_blur_focus_spread": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 8.0, "step": 0.1}),
                "radial_blur_steps": ("INT", {"default": 5, "min": 1, "max": 32, "step": 1}),

                # --- GROUP: Stylistic Effects ---
                "grp_stylistic": (["Stylistic Effects"], {}),
                "chromatic_aberration_r_x": ("FLOAT", {"default": 1.0, "min": -50.0, "max": 50.0, "step": 0.5}),
                "chromatic_aberration_r_y": ("FLOAT", {"default": 0.0, "min": -50.0, "max": 50.0, "step": 0.5}),
                "chromatic_aberration_b_x": ("FLOAT", {"default": -1.0, "min": -50.0, "max": 50.0, "step": 0.5}),
                "chromatic_aberration_b_y": ("FLOAT", {"default": 0.0, "min": -50.0, "max": 50.0, "step": 0.5}),
                "chromatic_blur_amount": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "simple_film_grain_intensity": ("FLOAT", {"default": 0.07, "min": 0.00, "max": 1.00, "step": 0.01}),
                "simple_film_grain_monochrome": ("BOOLEAN", {"default": True}),
                "scanline_intensity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 1.0}),
                "vignette_intensity": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 1.00, "step": 0.01}),
                "vignette_center_x": ("FLOAT", {"default": 0.50, "min": 0.00, "max": 1.00, "step": 0.01}),
                "vignette_center_y": ("FLOAT", {"default": 0.50, "min": 0.00, "max": 1.00, "step": 0.01}),
                "soft_light_opacity": ("FLOAT", {"default": 0.30, "min": 0.00, "max": 1.00, "step": 0.01}),
                "soft_light_blur_radius": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 50.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_filters"
    CATEGORY = "triXope/image"

    def _read_cube_file(self, filepath):
        lines = []
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    lines.append(line)

        lut_size = 0
        domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        table_data_lines = []
        
        for line_idx, line in enumerate(lines):
            if line.startswith('TITLE'): pass
            elif line.startswith('LUT_3D_SIZE'): 
                try: lut_size = int(line.split()[-1])
                except: raise ValueError(f"Error parsing LUT_3D_SIZE in {filepath}: '{line}'")
            elif line.startswith('DOMAIN_MIN'): 
                try: domain_min = np.array([float(x) for x in line.split()[1:]], dtype=np.float32)
                except: raise ValueError(f"Error parsing DOMAIN_MIN in {filepath}: '{line}'")
            elif line.startswith('DOMAIN_MAX'): 
                try: domain_max = np.array([float(x) for x in line.split()[1:]], dtype=np.float32)
                except: raise ValueError(f"Error parsing DOMAIN_MAX in {filepath}: '{line}'")
            else:
                try: 
                    parts = line.split()
                    if len(parts) == 3:
                        table_data_lines.append(list(map(float, parts)))
                    else:
                        print(f"Warning: Skipping malformed LUT data line {line_idx+1} in {filepath}: '{line}' (expected 3 values)")
                except ValueError: 
                    print(f"Warning: Could not parse LUT data line {line_idx+1} in {filepath}: '{line}'")
                    continue
        
        if lut_size == 0: raise ValueError(f"LUT_3D_SIZE not found or zero in {filepath}")
        if not table_data_lines: raise ValueError(f"No data lines found in LUT {filepath}")

        expected_lines = lut_size ** 3
        if len(table_data_lines) != expected_lines: 
            raise ValueError(f"LUT data lines ({len(table_data_lines)}) do not match LUT_3D_SIZE ({expected_lines}) in {filepath}")

        lut_table = np.array(table_data_lines, dtype=np.float32).reshape(lut_size, lut_size, lut_size, 3)
        return lut_table, lut_size, domain_min, domain_max

    def _apply_lut_to_image_scipy(self, image_np_rgb_0_1, lut_data):
        if not SCIPY_AVAILABLE:
            print("SciPy not available, LUT application will use slower pixel-by-pixel method.")
            return self._apply_lut_to_image_numpy_slow(image_np_rgb_0_1, lut_data)

        lut_table, lut_size, _, _ = lut_data
        r_axis = np.linspace(0, 1, lut_size)
        g_axis = np.linspace(0, 1, lut_size)
        b_axis = np.linspace(0, 1, lut_size)
        
        interp_R_out = RegularGridInterpolator((b_axis, g_axis, r_axis), lut_table[..., 0], bounds_error=False, fill_value=None)
        interp_G_out = RegularGridInterpolator((b_axis, g_axis, r_axis), lut_table[..., 1], bounds_error=False, fill_value=None)
        interp_B_out = RegularGridInterpolator((b_axis, g_axis, r_axis), lut_table[..., 2], bounds_error=False, fill_value=None)

        image_bgr_0_1 = image_np_rgb_0_1[..., [2, 1, 0]]
        points_to_interpolate = image_bgr_0_1.reshape(-1, 3)
        
        output_r = interp_R_out(points_to_interpolate)
        output_g = interp_G_out(points_to_interpolate)
        output_b = interp_B_out(points_to_interpolate)
        
        output_image_np = np.stack([output_r, output_g, output_b], axis=-1)
        output_image_np = output_image_np.reshape(image_np_rgb_0_1.shape)
        
        return np.clip(output_image_np, 0.0, 1.0)

    def _trilinear_interpolation_slow(self, p_rgb, lut_table, lut_size):
        r_norm, g_norm, b_norm = p_rgb 
        x = r_norm * (lut_size - 1)
        y = g_norm * (lut_size - 1)
        z = b_norm * (lut_size - 1)
        x0, y0, z0 = int(math.floor(x)), int(math.floor(y)), int(math.floor(z))
        x0 = np.clip(x0, 0, lut_size - 1)
        y0 = np.clip(y0, 0, lut_size - 1)
        z0 = np.clip(z0, 0, lut_size - 1)
        x1 = min(x0 + 1, lut_size - 1)
        y1 = min(y0 + 1, lut_size - 1)
        z1 = min(z0 + 1, lut_size - 1)
        xd, yd, zd = x - x0, y - y0, z - z0
        
        c000 = lut_table[z0, y0, x0]
        c001 = lut_table[z0, y0, x1]
        c010 = lut_table[z0, y1, x0]
        c011 = lut_table[z0, y1, x1]
        c100 = lut_table[z1, y0, x0]
        c101 = lut_table[z1, y0, x1]
        c110 = lut_table[z1, y1, x0]
        c111 = lut_table[z1, y1, x1]

        c00 = c000 * (1 - xd) + c001 * xd
        c01 = c010 * (1 - xd) + c011 * xd
        c10 = c100 * (1 - xd) + c101 * xd
        c11 = c110 * (1 - xd) + c111 * xd
        
        c0 = c00 * (1 - yd) + c01 * yd
        c1 = c10 * (1 - yd) + c11 * yd

        c_final = c0 * (1 - zd) + c1 * zd
        return c_final

    def _apply_lut_to_image_numpy_slow(self, image_np_rgb_0_1, lut_data):
        lut_table, lut_size, _, _ = lut_data
        output_image_np = np.zeros_like(image_np_rgb_0_1)
        for r_idx in range(image_np_rgb_0_1.shape[0]):
            for c_idx in range(image_np_rgb_0_1.shape[1]):
                pixel_val_rgb_0_1 = image_np_rgb_0_1[r_idx, c_idx]
                output_image_np[r_idx, c_idx] = self._trilinear_interpolation_slow(pixel_val_rgb_0_1, lut_table, lut_size)
        return np.clip(output_image_np, 0.0, 1.0)

    def _apply_lut_effect(self, pil_image, lut_name, strength, log_process):
        if strength == 0.0 or lut_name == "None": return pil_image
        try:
            lut_path = folder_paths.get_full_path("luts", lut_name)
            if not lut_path or not os.path.exists(lut_path):
                print(f"Warning: LUT file '{lut_name}' not found. Skipping LUT."); return pil_image
            lut_data = self._read_cube_file(lut_path)
        except Exception as e: print(f"Error reading LUT '{lut_name}': {e}. Skipping."); return pil_image

        image_np_0_1 = np.array(pil_image, dtype=np.float32) / 255.0
        original_image_np_0_1 = image_np_0_1.copy()

        _, _, domain_min, domain_max = lut_data
        dom_scale = domain_max - domain_min
        is_non_default_domain = not (np.allclose(domain_min, 0.0) and np.allclose(domain_max, 1.0))

        lut_input_normalized_0_1 = image_np_0_1.copy()
        if is_non_default_domain:
            for c_idx in range(3):
                if dom_scale[c_idx] != 0:
                    lut_input_normalized_0_1[..., c_idx] = (lut_input_normalized_0_1[..., c_idx] - domain_min[c_idx]) / dom_scale[c_idx]
            lut_input_normalized_0_1 = np.clip(lut_input_normalized_0_1, 0.0, 1.0)

        if log_process: lut_input_normalized_0_1 = np.power(np.clip(lut_input_normalized_0_1, 1e-5, 1.0), 1.0 / 2.2)
        
        lut_output_normalized_0_1 = self._apply_lut_to_image_scipy(lut_input_normalized_0_1, lut_data)

        if log_process: lut_output_normalized_0_1 = np.power(np.clip(lut_output_normalized_0_1, 0.0, 1.0), 2.2)

        final_lut_applied_np_0_1 = lut_output_normalized_0_1.copy()
        if is_non_default_domain:
            for c_idx in range(3):
                if dom_scale[c_idx] != 0:
                     final_lut_applied_np_0_1[..., c_idx] = final_lut_applied_np_0_1[..., c_idx] * dom_scale[c_idx] + domain_min[c_idx]
                else:
                     final_lut_applied_np_0_1[..., c_idx] = domain_min[c_idx] 
        final_lut_applied_np_0_1 = np.clip(final_lut_applied_np_0_1, 0.0, 1.0)

        blended_np_0_1 = (1.0 - strength) * original_image_np_0_1 + strength * final_lut_applied_np_0_1
        blended_np_0_1 = np.clip(blended_np_0_1, 0.0, 1.0)
        return Image.fromarray((blended_np_0_1 * 255).astype(np.uint8))

    def _apply_shadows_highlights(self, pil_image, shadow_adj, highlight_adj, hdr_intensity):
        if shadow_adj == 0.0 and highlight_adj == 0.0 and hdr_intensity == 1.0: return pil_image
        img_array = np.array(pil_image, dtype=np.float32) / 255.0
        if hdr_intensity != 1.0:
            mean = np.mean(img_array)
            img_array = mean + (img_array - mean) * (1.0 + (hdr_intensity - 1.0) * 0.5) 
        
        luminance = 0.299 * img_array[..., 0] + 0.587 * img_array[..., 1] + 0.114 * img_array[..., 2]
        luminance = np.clip(luminance, 0.0, 1.0)[..., np.newaxis]

        if shadow_adj > 0.0:
            shadow_map = 1.0 - luminance
            shadow_boost = shadow_map * shadow_adj
            img_array += shadow_boost
        if highlight_adj > 0.0:
            highlight_map = luminance
            highlight_reduction = highlight_map * highlight_adj
            img_array -= highlight_reduction
            
        img_array = np.clip(img_array, 0.0, 1.0)
        return Image.fromarray((img_array * 255).astype(np.uint8))

    def _apply_gamma_numpy(self, pil_image, gamma_value):
        if gamma_value == 1.0: return pil_image
        img_np = np.array(pil_image, dtype=np.float32) / 255.0
        gamma_correction = 1.0 / max(gamma_value, 0.01)
        corrected_np = np.power(np.clip(img_np, 1e-5, 1.0), gamma_correction)
        return Image.fromarray((np.clip(corrected_np * 255, 0, 255)).astype(np.uint8))

    def _apply_color_enhancements(self, pil_image, brightness, contrast, saturation, enhance_color_sat):
        if brightness!=1.: pil_image=ImageEnhance.Brightness(pil_image).enhance(brightness)
        if contrast!=1.: pil_image=ImageEnhance.Contrast(pil_image).enhance(contrast)
        cur_sat=saturation*enhance_color_sat
        if cur_sat!=1.: pil_image=ImageEnhance.Color(pil_image).enhance(cur_sat)
        return pil_image

    def _apply_sharpness_detail(self, pil_image, sharpness_factor, edge_enhance_str, detail_enhance_str):
        if sharpness_factor!=0.: pil_image=ImageEnhance.Sharpness(pil_image).enhance(1.+sharpness_factor)
        if edge_enhance_str>0.: pil_image=Image.blend(pil_image,pil_image.filter(ImageFilter.EDGE_ENHANCE_MORE),edge_enhance_str)
        if detail_enhance_str>0.: pil_image=Image.blend(pil_image,pil_image.filter(ImageFilter.DETAIL),detail_enhance_str)
        return pil_image

    def _apply_blurs(self, pil_image, blur_r, gaussian_blur_r):
        if blur_r>0: pil_image=pil_image.filter(ImageFilter.BoxBlur(blur_r))
        if gaussian_blur_r>0.: pil_image=pil_image.filter(ImageFilter.GaussianBlur(radius=gaussian_blur_r))
        return pil_image

    def _apply_chromatic_aberration(self, pil_image, r_x, r_y, b_x, b_y, blur_amount):
        if all(v==0 for v in [r_x,r_y,b_x,b_y]) and blur_amount==0.: return pil_image
        img=np.array(pil_image).astype(np.float32); h,w,_=img.shape
        rc,gc,bc = img[...,0],img[...,1],img[...,2]
        if blur_amount>0.: bk=int(blur_amount*2)*2+1; rc,bc = cv2.GaussianBlur(rc,(bk,bk),blur_amount),cv2.GaussianBlur(bc,(bk,bk),blur_amount)
        Mr,Mb = np.float32([[1,0,r_x],[0,1,r_y]]),np.float32([[1,0,b_x],[0,1,b_y]])
        rs,bs = cv2.warpAffine(rc,Mr,(w,h),borderMode=cv2.BORDER_REFLECT_101),cv2.warpAffine(bc,Mb,(w,h),borderMode=cv2.BORDER_REFLECT_101)
        return Image.fromarray(np.clip(cv2.merge([rs,gc,bs]),0,255).astype(np.uint8))

    def _apply_radial_blur(self, pil_image, strength, center_x_r, center_y_r, focus_spread, steps):
        if strength==0.: return pil_image
        img=np.array(pil_image).astype(np.float32)/255.; h,w,_=img.shape; cx,cy=int(w*center_x_r),int(h*center_y_r)
        max_d=np.sqrt(max(cx**2+cy**2,(w-cx)**2+cy**2,cx**2+(h-cy)**2,(w-cx)**2+(h-cy)**2))
        X,Y=np.meshgrid(np.arange(w)-cx,np.arange(h)-cy); rad_mask=np.sqrt(X**2+Y**2)/max(1.,max_d)
        blurred,cur_blur=[],strength
        for _ in range(steps):
            k=int(cur_blur); blurred.append(cv2.GaussianBlur(img,(k|1,k|1),0) if k>0 else img.copy()); cur_blur/=focus_spread
        final=np.zeros_like(img)
        if not blurred: final=img
        else:
            for i in range(steps):
                m_i=np.clip((rad_mask-i/steps)*steps,0,1); inv_m_i=1.-m_i
                if len(img.shape)==3: m_i,inv_m_i=[np.dstack([m]*3) for m in (m_i,inv_m_i)]
                final=blurred[steps-1-i]*m_i+(final if i>0 else img)*inv_m_i
        return Image.fromarray(np.clip(final*255,0,255).astype(np.uint8))

    def _apply_simple_film_grain(self, pil_image, intensity, monochrome):
        if intensity==0.: return pil_image
        img=np.array(pil_image,dtype=np.float32); h,w,c=img.shape
        noise=np.random.normal(0,intensity*127,(h,w,1 if monochrome else c))
        if monochrome and c==3: noise=np.repeat(noise,3,axis=2)
        return Image.fromarray(np.clip(img+noise,0,255).astype(np.uint8))

    def _apply_scanlines(self, pil_image, intensity):
        if intensity==0.: return pil_image
        img=np.array(pil_image); scan=np.ones(img.shape[:2],dtype=img.dtype); scan[::2,:]=1.-intensity
        if len(img.shape)==3: scan=np.dstack([scan]*img.shape[2])
        return Image.fromarray((img*scan).astype(np.uint8))

    def _apply_vignette(self, pil_image, intensity, center_x_r, center_y_r):
        if intensity==0.: return pil_image
        img=np.array(pil_image).astype(np.float32)/255.; h,w=img.shape[:2]
        cx,cy=w*center_x_r,h*center_y_r
        X,Y=np.meshgrid(np.arange(w)-cx,np.arange(h)-cy); dist=np.sqrt(X**2+Y**2)
        max_d=np.sqrt(max(cx**2+cy**2,(w-cx)**2+cy**2,cx**2+(h-cy)**2,(w-cx)**2+(h-cy)**2))
        v_mask=np.clip(1.-(dist/max(1.,max_d))**2*intensity,0.,1.)
        if len(img.shape)==3: v_mask=np.dstack([v_mask]*img.shape[2])
        return Image.fromarray((np.clip(img*v_mask,0,1)*255).astype(np.uint8))

    def _apply_soft_light(self, pil_image, opacity, blur_radius):
        if opacity==0.: return pil_image
        blur=pil_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        return Image.blend(pil_image,ImageChops.soft_light(pil_image,blur),opacity)

    def _apply_gamma_torch(self, tensor_images, gamma_value):
        if gamma_value == 1.0: return tensor_images
        gamma_correction = 1.0 / max(gamma_value, 0.01)
        return torch.pow(tensor_images.clamp(1e-5, 1.0), gamma_correction)

    def _apply_simple_film_grain_torch(self, tensor_images, intensity, monochrome):
        if intensity == 0.: return tensor_images
        B, H, W, C = tensor_images.shape
        noise_shape = (B, H, W, 1 if monochrome else C)
        noise = torch.randn(noise_shape, device=tensor_images.device) * intensity
        if monochrome and C > 1:
            noise = noise.repeat(1, 1, 1, C)
        return (tensor_images + noise).clamp(0.0, 1.0)

    def _apply_vignette_torch(self, tensor_images, intensity, center_x_r, center_y_r):
        if intensity == 0.: return tensor_images
        B, H, W, C = tensor_images.shape
        device = tensor_images.device
        
        cx, cy = W * center_x_r, H * center_y_r
        X = torch.arange(W, device=device) - cx
        Y = torch.arange(H, device=device) - cy
        yy, xx = torch.meshgrid(Y, X, indexing='ij')
        
        dist = torch.sqrt(xx**2 + yy**2)
        max_dist = torch.sqrt(torch.tensor(max(cx**2+cy**2,(W-cx)**2+cy**2,cx**2+(H-cy)**2,(W-cx)**2+(H-cy)**2), device=device))
        
        v_mask = (1.0 - (dist / max(1.0, max_dist))**2 * intensity).clamp(0.0, 1.0)
        v_mask = v_mask.unsqueeze(0).unsqueeze(-1)

        return (tensor_images * v_mask).clamp(0.0, 1.0)
    
    def process_filters(self, images, enable,
                        enable_color_correction, enable_lut_processing, enable_enhancements,
                        enable_blur_effects, enable_stylistic_effects,
                        lut_name, lut_strength, lut_log_process,
                        hdr_intensity, shadow_intensity, highlight_intensity, gamma,
                        brightness, contrast, saturation,
                        enhance_color, sharpness, edge_enhance_strength, detail_enhance_strength,
                        blur_radius, gaussian_blur_radius,
                        radial_blur_strength, radial_blur_center_x, radial_blur_center_y,
                        radial_blur_focus_spread, radial_blur_steps,
                        chromatic_aberration_r_x, chromatic_aberration_r_y,
                        chromatic_aberration_b_x, chromatic_aberration_b_y, chromatic_blur_amount,
                        simple_film_grain_intensity, simple_film_grain_monochrome,
                        scanline_intensity,
                        vignette_intensity, vignette_center_x, vignette_center_y,
                        soft_light_opacity, soft_light_blur_radius, **kwargs
                        ):

        if not enable:
            return (images,)

        low_vram_mode = True 

        if low_vram_mode:
            print("INFO: ColorFX running in Low VRAM mode.")
            processed_images_tensors = []

            iterable = images
            if TQDM_AVAILABLE:
                iterable = tqdm(images, desc="ColorFX Processing Frames")

            for single_image_tensor in iterable:
                pil_image = tensor2pil(single_image_tensor).convert("RGB")
                if enable_color_correction:
                    pil_image = self._apply_shadows_highlights(pil_image, shadow_intensity, highlight_intensity, hdr_intensity)
                    pil_image = self._apply_gamma_numpy(pil_image, gamma)
                    pil_image = self._apply_color_enhancements(pil_image, brightness, contrast, saturation, enhance_color)
                if enable_lut_processing and lut_name != "None":
                    pil_image = self._apply_lut_effect(pil_image, lut_name, lut_strength, lut_log_process)
                if enable_enhancements:
                    pil_image = self._apply_sharpness_detail(pil_image, sharpness, edge_enhance_strength, detail_enhance_strength)
                if enable_blur_effects:
                    pil_image = self._apply_blurs(pil_image, blur_radius, gaussian_blur_radius)
                    if radial_blur_strength > 0.0:
                        pil_image = self._apply_radial_blur(pil_image, radial_blur_strength, radial_blur_center_x, radial_blur_center_y, radial_blur_focus_spread, radial_blur_steps)
                if enable_stylistic_effects:
                    if any(v!=0 for v in [chromatic_aberration_r_x, chromatic_aberration_r_y, chromatic_aberration_b_x, chromatic_aberration_b_y]) or chromatic_blur_amount > 0:
                        pil_image = self._apply_chromatic_aberration(pil_image, chromatic_aberration_r_x, chromatic_aberration_r_y, chromatic_aberration_b_x, chromatic_aberration_b_y, chromatic_blur_amount)
                    if simple_film_grain_intensity > 0.0:
                        pil_image = self._apply_simple_film_grain(pil_image, simple_film_grain_intensity, simple_film_grain_monochrome)
                    if scanline_intensity > 0.0:
                        pil_image = self._apply_scanlines(pil_image, scanline_intensity)
                    if vignette_intensity > 0.0:
                        pil_image = self._apply_vignette(pil_image, vignette_intensity, vignette_center_x, vignette_center_y)
                    if soft_light_opacity > 0.0:
                         pil_image = self._apply_soft_light(pil_image, soft_light_opacity, soft_light_blur_radius)
                processed_images_tensors.append(pil2tensor(pil_image))
            return (torch.cat(processed_images_tensors, dim=0),)

        print("INFO: ColorFX running in Fast (batched) mode.")
        processed_tensor = images.clone()
        
        if enable_color_correction:
            pil_images = [tensor2pil(processed_tensor[i]).convert("RGB") for i in range(processed_tensor.shape[0])]
            processed_pils = []
            for pil_image in pil_images:
                img = self._apply_shadows_highlights(pil_image, shadow_intensity, highlight_intensity, hdr_intensity)
                img = self._apply_color_enhancements(img, brightness, contrast, saturation, enhance_color)
                processed_pils.append(img)
            processed_tensor = torch.cat([pil2tensor(p) for p in processed_pils], dim=0).to(images.device)
            processed_tensor = self._apply_gamma_torch(processed_tensor, gamma)

        if enable_lut_processing and lut_name != "None":
            pil_images = [tensor2pil(processed_tensor[i]).convert("RGB") for i in range(processed_tensor.shape[0])]
            processed_pils = [self._apply_lut_effect(p, lut_name, lut_strength, lut_log_process) for p in pil_images]
            processed_tensor = torch.cat([pil2tensor(p) for p in processed_pils], dim=0).to(images.device)

        if enable_enhancements or enable_blur_effects:
            pil_images = [tensor2pil(processed_tensor[i]).convert("RGB") for i in range(processed_tensor.shape[0])]
            processed_pils = []
            for pil_image in pil_images:
                if enable_enhancements:
                    pil_image = self._apply_sharpness_detail(pil_image, sharpness, edge_enhance_strength, detail_enhance_strength)
                if enable_blur_effects:
                    pil_image = self._apply_blurs(pil_image, blur_radius, gaussian_blur_radius)
                    if radial_blur_strength > 0.0:
                        pil_image = self._apply_radial_blur(pil_image, radial_blur_strength, radial_blur_center_x, radial_blur_center_y, radial_blur_focus_spread, radial_blur_steps)
                processed_pils.append(pil_image)
            processed_tensor = torch.cat([pil2tensor(p) for p in processed_pils], dim=0).to(images.device)
            
        if enable_stylistic_effects:
            needs_pil_path = any(v!=0 for v in [chromatic_aberration_r_x, chromatic_aberration_r_y, chromatic_aberration_b_x, chromatic_aberration_b_y]) \
                             or chromatic_blur_amount > 0 or scanline_intensity > 0.0 or soft_light_opacity > 0.0

            if needs_pil_path:
                pil_images = [tensor2pil(processed_tensor[i]).convert("RGB") for i in range(processed_tensor.shape[0])]
                processed_pils = []
                for pil_image in pil_images:
                    if any(v!=0 for v in [chromatic_aberration_r_x, chromatic_aberration_r_y, chromatic_aberration_b_x, chromatic_aberration_b_y]) or chromatic_blur_amount > 0:
                        pil_image = self._apply_chromatic_aberration(pil_image, chromatic_aberration_r_x, chromatic_aberration_r_y, chromatic_aberration_b_x, chromatic_aberration_b_y, chromatic_blur_amount)
                    if scanline_intensity > 0.0:
                        pil_image = self._apply_scanlines(pil_image, scanline_intensity)
                    if soft_light_opacity > 0.0:
                        pil_image = self._apply_soft_light(pil_image, soft_light_opacity, soft_light_blur_radius)
                    processed_pils.append(pil_image)
                processed_tensor = torch.cat([pil2tensor(p) for p in processed_pils], dim=0).to(images.device)
            
            if simple_film_grain_intensity > 0.0:
                processed_tensor = self._apply_simple_film_grain_torch(processed_tensor, simple_film_grain_intensity, simple_film_grain_monochrome)
            if vignette_intensity > 0.0:
                processed_tensor = self._apply_vignette_torch(processed_tensor, vignette_intensity, vignette_center_x, vignette_center_y)

        return (processed_tensor,)

# --- moved from color_match.py ----------------------------------------------
import torch
import numpy as np
import cv2

class HistogramMatchNode:
    """
    A node to match the contrast and saturation of a target image to a reference image
    by performing histogram matching on the Saturation and Value channels in the HSV color space.
    This version supports batch processing.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "mix_factor": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"
                }),
                "match_hue": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "match_histograms"
    CATEGORY = "triXope/image"

    def _match_channel_histogram(self, source_channel, template_channel):
        """
        Matches the histogram of a source channel to a template channel.
        Returns the modified source channel.
        """
        # Calculate histograms and CDFs for both channels
        source_hist = cv2.calcHist([source_channel], [0], None, [256], [0, 256])
        template_hist = cv2.calcHist([template_channel], [0], None, [256], [0, 256])
        source_cdf = source_hist.cumsum()
        template_cdf = template_hist.cumsum()

        # Normalize CDFs
        source_cdf_normalized = source_cdf / source_cdf.max()
        template_cdf_normalized = template_cdf / template_cdf.max()
        
        # Create the lookup table (LUT)
        lookup_table = np.zeros(256, np.uint8)
        g = 0
        for f in range(256):
            while g < 256 and template_cdf_normalized[g] < source_cdf_normalized[f]:
                g += 1
            lookup_table[f] = g
            
        return cv2.LUT(source_channel, lookup_table)

    def match_histograms(self, reference_image, target_image, mix_factor, match_hue):
        # 1. Prepare the reference image (using the first image in its batch)
        ref_tensor = reference_image[0]
        ref_np_rgb = (ref_tensor.cpu().numpy() * 255).astype(np.uint8)
        ref_hsv = cv2.cvtColor(ref_np_rgb, cv2.COLOR_RGB2HSV)
        h_ref, s_ref, v_ref = cv2.split(ref_hsv)

        processed_images = []

        # 2. Loop through each image in the target batch
        for i in range(target_image.shape[0]):
            tar_tensor = target_image[i]
            tar_np_rgb_original = (tar_tensor.cpu().numpy() * 255).astype(np.uint8)
            
            tar_hsv = cv2.cvtColor(tar_np_rgb_original, cv2.COLOR_RGB2HSV)
            h_tar, s_tar, v_tar = cv2.split(tar_hsv)

            # 3. Perform histogram matching on Saturation and Value channels
            s_matched = self._match_channel_histogram(s_tar, s_ref)
            v_matched = self._match_channel_histogram(v_tar, v_ref)

            # 4. Optionally match hue by shifting the mean
            h_final = h_tar
            if match_hue:
                h_ref_mean = int(cv2.mean(h_ref)[0])
                h_tar_mean = int(cv2.mean(h_tar)[0])
                hue_diff = h_ref_mean - h_tar_mean
                h_final = (h_tar.astype(np.int32) + hue_diff) % 180
                h_final = h_final.astype(np.uint8)

            # 5. Merge channels and convert back to RGB
            matched_hsv = cv2.merge([h_final, s_matched, v_matched])
            matched_rgb = cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
            
            # 6. Handle the mix factor
            if mix_factor < 1.0:
                matched_rgb = cv2.addWeighted(matched_rgb, mix_factor, tar_np_rgb_original, 1 - mix_factor, 0)
            
            processed_images.append(matched_rgb)

        # 7. Stack and convert the final batch to a tensor
        output_np_batch = np.array(processed_images)
        output_tensor = torch.from_numpy(output_np_batch.astype(np.float32) / 255.0)

        return (output_tensor,)

NODE_CLASS_MAPPINGS = {
    "HistogramMatchHSV": HistogramMatchNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HistogramMatchHSV": "triXope HSV 🎨"
}

# --- moved from image_batch_upscaler.py -------------------------------------
import torch
import math
from comfy.utils import ProgressBar, common_upscale
import comfy.model_management as model_management
from nodes import MAX_RESOLUTION

class ImageUpscaleToSize:
    """
    This node scales an image to a target size using an upscale model.
    It first downscales the image so that the upscale model's scaling factor
    results in the final desired dimensions. This allows resizing *using* the
    model's algorithm instead of a standard scaler.
    """
    resize_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "images": ("IMAGE",),
                "model_scale_factor": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 64.0, "step": 0.1, "doc": "The multiplier of the upscale model (e.g., 4.0 for a 4x model)."}),
                "final_width": ("INT", {"default": 1024, "min": 0, "max": MAX_RESOLUTION, "step": 8, "label": "Final Target Width"}),
                "final_height": ("INT", {"default": 1024, "min": 0, "max": MAX_RESOLUTION, "step": 8, "label": "Final Target Height"}),
                "keep_proportion": (["stretch", "crop", "pad"], {"default": "stretch"}),
                "downscale_method": (s.resize_methods, {"default": "lanczos"}),
                "crop_position": (["center", "top", "bottom", "left", "right"], {"default": "center"}),
                "pad_color": ("STRING", {"default": "0, 0, 0", "tooltip": "RGB color for padding, e.g., '0, 0, 0' for black."}),
                "per_batch": ("INT", {"default": 8, "min": 1, "max": 4096, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale_to_size"
    CATEGORY = "triXope/image"

    def upscale_to_size(self, upscale_model, images, model_scale_factor, final_width, final_height, keep_proportion, downscale_method, crop_position, pad_color, per_batch):
        if model_scale_factor <= 0:
            raise ValueError("Model Scale Factor must be greater than zero.")

        device = model_management.get_torch_device()
        upscale_model.to(device)

        in_img_batch = images.movedim(-1, 1) # B,C,H,W

        pbar = ProgressBar(in_img_batch.shape[0])
        upscaled_batches = []

        for i in range(0, in_img_batch.shape[0], per_batch):
            sub_batch = in_img_batch[i:i + per_batch]
            _B, _C, H, W = sub_batch.shape

            target_width = final_width if final_width > 0 else round(W * model_scale_factor)
            target_height = final_height if final_height > 0 else round(H * model_scale_factor)

            # Calculate the intermediate dimensions needed before applying the model
            intermediate_width = math.ceil(target_width / model_scale_factor)
            intermediate_height = math.ceil(target_height / model_scale_factor)

            pre_scaled_batch = sub_batch

            if keep_proportion == "stretch":
                pre_scaled_batch = common_upscale(sub_batch, intermediate_width, intermediate_height, downscale_method, "disabled")

            elif keep_proportion == "crop":
                original_aspect = W / H
                target_aspect = target_width / target_height

                if original_aspect > target_aspect:
                    crop_w = int(H * target_aspect)
                    crop_h = H
                else:
                    crop_w = W
                    crop_h = int(W / target_aspect)

                if crop_position == "top": y_start, x_start = 0, (W - crop_w) // 2
                elif crop_position == "bottom": y_start, x_start = H - crop_h, (W - crop_w) // 2
                elif crop_position == "left": y_start, x_start = (H - crop_h) // 2, 0
                elif crop_position == "right": y_start, x_start = (H - crop_h) // 2, W - crop_w
                else: y_start, x_start = (H - crop_h) // 2, (W - crop_w) // 2

                cropped = sub_batch[:, :, y_start:y_start+crop_h, x_start:x_start+crop_w]
                pre_scaled_batch = common_upscale(cropped, intermediate_width, intermediate_height, downscale_method, "disabled")

            elif keep_proportion == "pad":
                ratio = min(intermediate_width / W, intermediate_height / H)
                new_width, new_height = round(W * ratio), round(H * ratio)

                temp_resized = common_upscale(sub_batch, new_width, new_height, downscale_method, "disabled")

                color_vals = [float(x.strip()) / 255.0 for x in pad_color.split(',')]
                padded_batch = torch.zeros((_B, _C, intermediate_height, intermediate_width), dtype=temp_resized.dtype, device=temp_resized.device)
                for c in range(_C):
                    padded_batch[:, c, :, :] = color_vals[c % len(color_vals)]

                pad_top = (intermediate_height - new_height) // 2
                pad_left = (intermediate_width - new_width) // 2

                padded_batch[:, :, pad_top:pad_top+new_height, pad_left:pad_left+new_width] = temp_resized
                pre_scaled_batch = padded_batch
            
            # --- UPSCALE THE PRE-SCALED BATCH ---
            upscaled_sub_batch = upscale_model(pre_scaled_batch.to(device))
            upscaled_batches.append(upscaled_sub_batch.cpu())
            
            pbar.update(upscaled_sub_batch.shape[0])

        upscale_model.cpu()
        
        final_output = torch.cat(upscaled_batches, dim=0).permute(0, 2, 3, 1).cpu()

        return (final_output,)

# --- moved from image_batch_selector.py -------------------------------------
import torch

class ImageBatchSelector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 999, "step": 1}),
                "skip_first_images": ("INT", {"default": 0, "min": 0, "max": 999, "step": 1}),
                "inverse_selection": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "filter_batch"
    CATEGORY = "triXope/image"

    def filter_batch(self, images, select_every_nth, skip_first_images, inverse_selection):
        # 1. Skip the initial frames
        total_images = images.shape[0]
        if skip_first_images >= total_images:
            # Return a single black frame or empty if skip is too large
            # Better to return the last frame to avoid crashing most workflows
            return (images[-1:],)

        remaining_images = images[skip_first_images:]
        num_remaining = remaining_images.shape[0]
        
        # 2. Determine indices
        indices = list(range(0, num_remaining, select_every_nth))
        
        if inverse_selection:
            # Create a set of all indices and remove the 'every nth' ones
            all_indices = set(range(num_remaining))
            selected_set = set(indices)
            final_indices = sorted(list(all_indices - selected_set))
        else:
            final_indices = indices

        # 3. Guard against empty selection
        if not final_indices:
            return (images[-1:],)

        # 4. Extract the selected images
        output_batch = remaining_images[final_indices]
        
        return (output_batch,)

# --- moved from pano360.py --------------------------------------------------
import os
import shutil
import subprocess
import tempfile
import numpy as np
import torch
from PIL import Image

def get_ffmpeg_path():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    for name in ["ffmpeg", "ffmpeg.exe"]:
        path = shutil.which(name)
        if path:
            return path

    raise RuntimeError(
        "FFmpeg executable not found. Make sure FFmpeg is in your system PATH "
        "or imageio-ffmpeg is installed in your ComfyUI environment."
    )

class Pano360Rotate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {
                    "default": 4096,
                    "min": 256,
                    "max": 8192,
                    "step": 64,
                    "display": "slider",
                }),
                "height": ("INT", {
                    "default": 2048,
                    "min": 128,
                    "max": 4096,
                    "step": 64,
                    "display": "slider",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "triXope/image"

    def process(self, image: torch.Tensor, width: int, height: int):
        img_tensor = image[0].detach().cpu()
        if img_tensor.shape[-1] == 4:
            img_tensor = img_tensor[:, :, :3]

        img_np = (img_tensor.numpy() * 255.0).clip(0, 255).astype(np.uint8)

        ffmpeg_bin = get_ffmpeg_path()
        frame_bytes = width * height * 3

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input = os.path.join(temp_dir, "input.png")
            Image.fromarray(img_np).save(temp_input)

            cmd = [
                ffmpeg_bin,
                "-y",
                "-loop", "1",
                "-i", temp_input,
                "-vf", f"scroll=horizontal=0.025,v360=input=equirect:output=flat:h_fov=90:v_fov=90:w={width}:h={height}",
                "-t", "2",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-"
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout_data, stderr_data = process.communicate()

            if process.returncode != 0:
                err_msg = stderr_data.decode("utf-8", errors="replace")
                raise RuntimeError(f"FFmpeg execution failed:\n{err_msg}")

            if len(stdout_data) < frame_bytes:
                raise RuntimeError("FFmpeg completed without generating output frames.")

            num_frames = len(stdout_data) // frame_bytes
            valid_bytes = num_frames * frame_bytes
            raw_array = np.frombuffer(
                stdout_data[:valid_bytes], dtype=np.uint8
            ).reshape((num_frames, height, width, 3))

            output_tensor = torch.from_numpy(raw_array.copy()).float().div_(255.0)

            return (output_tensor,)

# --- moved from nodes.py:PreviewAnimation -----------------------------------
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
    CATEGORY = "triXope/image"

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

__all__ = [
    "ColorFX",
    "HistogramMatchNode",
    "ImageUpscaleToSize",
    "ImageBatchSelector",
    "Pano360Rotate",
    "PreviewAnimation",
]
