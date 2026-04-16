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
                "images": ("IMAGE",),
                "low_vram_mode": ("BOOLEAN", {"default": False}),
                "enable_color_correction": ("BOOLEAN", {"default": True}),
                "enable_lut_processing": ("BOOLEAN", {"default": True}),
                "enable_enhancements": ("BOOLEAN", {"default": True}),
                "enable_blur_effects": ("BOOLEAN", {"default": True}),
                "enable_stylistic_effects": ("BOOLEAN", {"default": True}),

                "lut_name": (lut_files, {"default": "None"}),
                "lut_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "lut_log_process": ("BOOLEAN", {"default": True}),

                "hdr_intensity": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "shadow_intensity": ("FLOAT", {"default": 0.10, "min": 0.00, "max": 2.00, "step": 0.01}),
                "highlight_intensity": ("FLOAT", {"default": 0.20, "min": 0.00, "max": 2.00, "step": 0.01}),
                "gamma": ("FLOAT", {"default": 1.00, "min": 0.10, "max": 5.00, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 0.90, "min": 0.00, "max": 3.00, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 0.75, "min": 0.00, "max": 3.00, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 0.90, "min": 0.00, "max": 3.00, "step": 0.01}),

                "enhance_color": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 3.00, "step": 0.01}),
                "sharpness": ("FLOAT", {"default": 3.0, "min": -2.0, "max": 5.0, "step": 0.1}),
                "edge_enhance_strength": ("FLOAT", {"default": 0.00, "min": 0.00, "max": 1.00, "step": 0.01}),
                "detail_enhance_strength": ("FLOAT", {"default": 0.20, "min": 0.00, "max": 1.00, "step": 0.01}),

                "blur_radius": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "gaussian_blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 64.0, "step": 0.1}),
                "radial_blur_strength": ("FLOAT", {"default": 32.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "radial_blur_center_x": ("FLOAT", {"default": 0.50, "min": 0.00, "max": 1.00, "step": 0.01}),
                "radial_blur_center_y": ("FLOAT", {"default": 0.50, "min": 0.00, "max": 1.00, "step": 0.01}),
                "radial_blur_focus_spread": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 8.0, "step": 0.1}),
                "radial_blur_steps": ("INT", {"default": 5, "min": 1, "max": 32, "step": 1}),

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
    CATEGORY = "triXope"

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

    # --- NEW TORCH-BASED HELPER FUNCTIONS ---
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
        max_dist = torch.sqrt(torch.tensor(max(cx**2+cy**2,(W-cx)**2+cy**2,cx**2+(H-cy)**2,(w-cx)**2+(h-cy)**2), device=device))
        
        v_mask = (1.0 - (dist / max(1.0, max_dist))**2 * intensity).clamp(0.0, 1.0)
        v_mask = v_mask.unsqueeze(0).unsqueeze(-1)

        return (tensor_images * v_mask).clamp(0.0, 1.0)
    
    # --- MODIFIED: Main processing function with progress bar ---
    def process_filters(self, images,
                        low_vram_mode,
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
                        soft_light_opacity, soft_light_blur_radius
                        ):
        
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