import sys
import os
import cv2
import torch
import torch.nn.functional as F
import requests
import zipfile
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url

def download_codeformer_folder():
    """Download and extract CodeFormer folder from Dropbox"""
    url = "https://www.dropbox.com/scl/fo/cklg1pstgalzz8u2dlwa6/ACR-OXkqPX_kbvWQkLVpaSg?rlkey=78h20pli33tpw3194jpf6h1df&dl=1"
    
    if not os.path.exists('CodeFormer'):
        try:
            print("Downloading CodeFormer folder...")
            
            # Download the zip file
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Save the zip file
            with open('codeformer.zip', 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract the zip file
            with zipfile.ZipFile('codeformer.zip', 'r') as zip_ref:
                zip_ref.extractall('.')
            
            # Remove the zip file
            os.remove('codeformer.zip')
            
            print("CodeFormer folder downloaded and extracted successfully")
            
        except Exception as e:
            print(f"Error downloading CodeFormer folder: {str(e)}")
            raise e

# set enhancer with RealESRGAN
def set_realesrgan():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    half = True if torch.cuda.is_available() else False
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4
    )
    upsampler = RealESRGANer(
        scale=4,
        model_path='CodeFormer/CodeFormer/weights/realesrgan/RealESRGAN_x4plus.pth',
        model=model,
        tile=400,
        tile_pad=40,
        pre_pad=0,
        half=half,
        device=device,
    )
    return upsampler


def face_restoration(img, background_enhance, face_upsample, upscale, codeformer_fidelity, upsampler, codeformer_net, device):
    """Run a single prediction on the model"""
    # Download CodeFormer folder if it doesn't exist
    if not os.path.exists('CodeFormer'):
        download_codeformer_folder()
    
    # Add CodeFormer to system path
    if 'CodeFormer/CodeFormer' not in sys.path:
        sys.path.append('./CodeFormer/CodeFormer')

    try:
        # take the default setting for the demo
        has_aligned = False
        only_center_face = False
        draw_box = False
        detection_model = "retinaface_resnet50"

        background_enhance = background_enhance if background_enhance is not None else True
        face_upsample = face_upsample if face_upsample is not None else True
        upscale = upscale if (upscale is not None and upscale > 0) else 2

        upscale = int(upscale) # convert type to int
        if upscale > 4: # avoid memory exceeded due to too large upscale
            upscale = 4 
        if upscale > 2 and max(img.shape[:2])>1000: # avoid memory exceeded due to too large img resolution
            upscale = 2 
        if max(img.shape[:2]) > 1500: # avoid memory exceeded due to too large img resolution
            upscale = 1
            background_enhance = False
            face_upsample = False

        face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=detection_model,
            save_ext="png",
            use_parse=True,
        )
        bg_upsampler = upsampler if background_enhance else None
        face_upsampler = upsampler if face_upsample else None

        if has_aligned:
            # the input faces are already cropped and aligned
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_helper.is_gray = is_gray(img, threshold=5)
            face_helper.cropped_faces = [img]
        else:
            face_helper.read_image(img)
            # get face landmarks for each face
            num_det_faces = face_helper.get_face_landmarks_5(
            only_center_face=only_center_face, resize=640, eye_dist_threshold=5
            )
            # align and warp each face
            face_helper.align_warp_face()

        # face restoration for each cropped face
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(
                cropped_face / 255.0, bgr2rgb=True, float32=True
            )
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    output = codeformer_net(
                        cropped_face_t, w=codeformer_fidelity, adain=True
                    )[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except RuntimeError as error:
                print(f"Failed inference for CodeFormer: {error}")
                restored_face = tensor2img(
                    cropped_face_t, rgb2bgr=True, min_max=(-1, 1)
                )

            restored_face = restored_face.astype("uint8")
            face_helper.add_restored_face(restored_face)

        # paste_back
        if not has_aligned:
            # upsample the background
            if bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
            else:
                bg_img = None
            face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            if face_upsample and face_upsampler is not None:
                restored_img = face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img,
                    draw_box=draw_box,
                    face_upsampler=face_upsampler,
                )
            else:
                restored_img = face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img, draw_box=draw_box
                )

        restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
        return restored_img
    except Exception as error:
        print('Global exception', error)
        return None, None

def download_codeformer_model():
    """Download the CodeFormer model weights."""
    # Create the correct directory structure
    os.makedirs('codeformer/codeformer/weights/codeformer', exist_ok=True)
    
    url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
    output_path = 'codeformer/codeformer/weights/codeformer/codeformer.pth'
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("CodeFormer model downloaded successfully")
    except Exception as e:
        print(f"Error downloading CodeFormer model: {str(e)}")
