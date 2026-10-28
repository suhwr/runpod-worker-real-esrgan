import os
import io
import uuid
import base64
import cv2
import traceback
import requests
import torch
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.modules.rp_logger import RunPodLogger
from basicsr.archs.rrdbnet_arch import RRDBNet
from PIL import Image
from schemas.input import INPUT_SCHEMA

# Use /workspace for the Docker volume mount path
# otherwise use the current working directory
worker_build_env = os.getenv('WORKER_BUILD_ENV')
if worker_build_env:
    VOLUME_PATH = '/workspace'
else:
    VOLUME_PATH = os.getcwd()

GPU_ID = 0
TMP_PATH = f'{VOLUME_PATH}/tmp'
MODELS_PATH = f'{VOLUME_PATH}/models/ESRGAN'
GFPGAN_MODEL_PATH = f'{VOLUME_PATH}/models/GFPGAN/GFPGANv1.3.pth'

logger = RunPodLogger()

MAX_CONCURRENCY = max(1, int(os.getenv("MAX_CONCURRENCY", "1")))
DEFAULT_CATBOX_USER_HASH = os.getenv("CATBOX_USER_HASH", "0f9dc6f11f2f5ac4241235974")

# Global caches to prevent model reloading on every inference
UPSAMPLER_CACHE = {}
GFPGAN_CACHE = {}


def _upload_uguu(file_path):
    with open(file_path, "rb") as f:
        resp = requests.post(
            "https://uguu.se/upload",
            params={"output": "json"},
            files={"files[]": (os.path.basename(file_path), f)},
            timeout=45,
        )
        resp.raise_for_status()
        data = resp.json()
        files = data.get("files")
        if files and len(files) > 0 and files[0].get("url"):
            return files[0]["url"]
        raise ValueError(f"Uguu response invalid: {data}")


def _upload_catbox(file_path, user_hash=None):
    data = {"reqtype": "fileupload"}
    hash_to_use = user_hash or DEFAULT_CATBOX_USER_HASH
    if hash_to_use:
        data["userhash"] = hash_to_use

    with open(file_path, "rb") as f:
        resp = requests.post(
            "https://catbox.moe/user/api.php",
            data=data,
            files={"fileToUpload": (os.path.basename(file_path), f)},
            timeout=60,
        )
        resp.raise_for_status()
        url = resp.text.strip()
        if url.startswith("https://"):
            return url
        raise ValueError(f"Catbox failed with response: {url}")


def _upload_ucarecdn(file_path, pub_key="demopublickey"):
    with open(file_path, "rb") as f:
        resp = requests.post(
            "https://upload.uploadcare.com/base/",
            data={
                "UPLOADCARE_PUB_KEY": pub_key,
                "UPLOADCARE_STORE": "1",
            },
            files={"file": (os.path.basename(file_path), f)},
            timeout=60,
        )
        resp.raise_for_status()
        file_id = resp.json().get("file")
        if file_id:
            return f"https://ucarecdn.com/{file_id}/{os.path.basename(file_path)}"
        raise ValueError(f"Uploadcare failed: {resp.text}")


def upload_with_fallback(file_path, catbox_hash=None):
    errors = []
    try:
        return _upload_uguu(file_path), "uguu"
    except Exception as exc:
        errors.append(f"uguu: {exc}")
    try:
        return _upload_catbox(file_path, user_hash=catbox_hash), "catbox"
    except Exception as exc:
        errors.append(f"catbox: {exc}")
    try:
        return _upload_ucarecdn(file_path), "ucarecdn"
    except Exception as exc:
        errors.append(f"ucarecdn: {exc}")
    raise RuntimeError("All uploaders failed: " + " | ".join(errors))


# ---------------------------------------------------------------------------- #
# Adaptive Tiling Helper                                                       #
# ---------------------------------------------------------------------------- #
def calculate_adaptive_tile(w, h, user_tile=0):
    """
    Determines ideal tile size based on image resolution to balance
    speed, visual seamlessness, and VRAM stability.
    - If user specifies a positive tile, cap at 1024 to prevent memory blowup.
    - Standard images (<= 2.5M pixels, ~1920x1280): tile=0 (1 pass, blazing fast).
    - Large images (2.5M - 8M pixels, ~3200x2500): tile=1024 (few tiles, high speed, OOM-safe).
    - Huge images (> 8M pixels, e.g. 4864x3328 = 16.2M): tile=768 (strictly bounds VRAM < 2.5GB).
    """
    total_pixels = w * h
    if user_tile and user_tile > 0:
        return min(user_tile, 1024)

    if total_pixels <= 2_500_000:
        return 0
    elif total_pixels <= 8_000_000:
        return 1024
    else:
        return 768


# ---------------------------------------------------------------------------- #
# Model Loading & Caching                                                      #
# ---------------------------------------------------------------------------- #
def get_upsampler(model_name, netscale, model, model_path, tile, tile_pad, pre_pad, half):
    from realesrgan import RealESRGANer
    cache_key = (model_name, netscale, half)
    if cache_key not in UPSAMPLER_CACHE:
        logger.info(f"Loading {model_name} (half={half}) into VRAM cache...")
        UPSAMPLER_CACHE[cache_key] = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=None,
            model=model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=half,
            gpu_id=GPU_ID
        )
    upsampler = UPSAMPLER_CACHE[cache_key]
    upsampler.tile_size = tile
    upsampler.tile_pad = tile_pad
    upsampler.pre_pad = pre_pad
    return upsampler


def get_face_enhancer(outscale, upsampler):
    if outscale not in GFPGAN_CACHE:
        logger.info("Loading GFPGAN into VRAM cache...")
        from gfpgan import GFPGANer
        GFPGAN_CACHE[outscale] = GFPGANer(
            model_path=GFPGAN_MODEL_PATH,
            upscale=outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler
        )
    else:
        GFPGAN_CACHE[outscale].bg_upsampler = upsampler
    return GFPGAN_CACHE[outscale]


def upscale(
        source_image_path,
        image_extension,
        model_name='RealESRGAN_x4plus',
        outscale=4,
        face_enhance=False,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        denoise_strength=0.5
):
    model_name = model_name.split('.')[0]

    if image_extension == '.jpg':
        image_format = 'JPEG'
    elif image_extension == '.png':
        image_format = 'PNG'
    else:
        raise ValueError('Unsupported image type, must be either JPEG or PNG')

    if model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'RealESRNet_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
    elif model_name == '4x-UltraSharp':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'lollypop':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    else:
        raise ValueError(f'Unsupported model: {model_name}')

    model_path = os.path.join(MODELS_PATH, model_name + '.pth')
    if not os.path.isfile(model_path):
        raise Exception(f'Could not find model: {model_path}')

    img = cv2.imread(source_image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f'Source image ({source_image_path}) is corrupt')

    h, w = img.shape[:2]
    total_pixels = w * h

    # Safety ceiling for extreme resolution images (> 25M pixels)
    MAX_PIXELS = 25_000_000
    if total_pixels > MAX_PIXELS:
        logger.info(f"Image has {total_pixels} pixels, resizing within {MAX_PIXELS} ceiling")
        scale_factor = (MAX_PIXELS / total_pixels) ** 0.5
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w

    # Determine adaptive tile
    effective_tile = calculate_adaptive_tile(w, h, user_tile=tile)
    effective_half = half if half is not None else torch.cuda.is_available()

    logger.info(f"Processing image {w}x{h} ({w*h} px) with tile={effective_tile}, tile_pad={tile_pad}, half={effective_half}")

    upsampler = get_upsampler(
        model_name=model_name,
        netscale=netscale,
        model=model,
        model_path=model_path,
        tile=effective_tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=effective_half
    )

    if face_enhance:
        face_enhancer = get_face_enhancer(outscale, upsampler)

    try:
        if face_enhance:
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as e:
        raise RuntimeError(e)
    else:
        result_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        output_buffer = io.BytesIO()
        result_image.save(output_buffer, format=image_format)
        image_data = output_buffer.getvalue()

        # If PNG and > 7MB, try compressing to high quality JPEG to save payload size
        if len(image_data) > 7 * 1024 * 1024 and image_format == 'PNG':
            jpeg_buffer = io.BytesIO()
            result_image.save(jpeg_buffer, format='JPEG', quality=95)
            if len(jpeg_buffer.getvalue()) <= 7 * 1024 * 1024:
                image_data = jpeg_buffer.getvalue()
                image_format = 'JPEG'

        return image_data, image_format


def determine_file_extension(image_data):
    try:
        if image_data.startswith('/9j/'):
            return '.jpg'
        elif image_data.startswith('iVBORw0Kg'):
            return '.png'
        else:
            return '.png'
    except Exception:
        return '.png'


def _save_source_image(source_image_b64, target_path):
    source_bytes = base64.b64decode(source_image_b64)
    with open(target_path, 'wb') as f:
        f.write(source_bytes)


# ---------------------------------------------------------------------------- #
# Serverless Handler (Sequential Hardcoded Queue, Concurrency=1)               #
# ---------------------------------------------------------------------------- #
def handler(event):
    if not os.path.exists(TMP_PATH):
        os.makedirs(TMP_PATH, exist_ok=True)

    validated_input = validate(event['input'], INPUT_SCHEMA)
    if 'errors' in validated_input:
        return {'errors': validated_input['errors']}

    input_data = validated_input['validated_input']
    unique_id = uuid.uuid4()
    source_image_data = input_data['source_image']
    source_file_extension = determine_file_extension(source_image_data)
    source_image_path = f'{TMP_PATH}/source_{unique_id}{source_file_extension}'

    _save_source_image(source_image_data, source_image_path)

    try:
        image_data, image_format = upscale(
            source_image_path,
            source_file_extension,
            input_data['model'],
            input_data['scale'],
            input_data['face_enhance'],
            input_data.get('tile', 0),
            input_data.get('tile_pad', 10),
            input_data.get('pre_pad', 0),
            input_data.get('half', False)
        )
    except Exception as e:
        logger.error(f'An exception was raised: {e}')
        return {
            'error': traceback.format_exc(),
            'refresh_worker': True
        }
    finally:
        if os.path.exists(source_image_path):
            try:
                os.remove(source_image_path)
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Handle large payload / upload fallback
    if len(image_data) > 7 * 1024 * 1024:
        ext = '.jpg' if image_format == 'JPEG' else '.png'
        out_filename = f'upscaled_{unique_id}{ext}'
        temp_out_path = f'{TMP_PATH}/{out_filename}'
        with open(temp_out_path, 'wb') as f:
            f.write(image_data)
        try:
            uploaded_url, provider = upload_with_fallback(temp_out_path)
            return {
                'image': '',
                'image_url': uploaded_url,
                'provider': provider
            }
        except Exception as upload_err:
            logger.error(f'Upload fallback failed: {upload_err}')
        finally:
            if os.path.exists(temp_out_path):
                try:
                    os.remove(temp_out_path)
                except Exception:
                    pass

    return {
        'image': base64.b64encode(image_data).decode('utf-8')
    }


def concurrency_controller(current_concurrency):
    return MAX_CONCURRENCY


if __name__ == "__main__":
    logger.info(f"Starting Serverless Worker (Max Concurrency: {MAX_CONCURRENCY}, Adaptive Tiling)...")
    start_params = {
        'handler': handler
    }
    if MAX_CONCURRENCY > 1:
        start_params['concurrency_modifier'] = concurrency_controller

    runpod.serverless.start(start_params)
