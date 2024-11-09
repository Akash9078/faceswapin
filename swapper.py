import os
import cv2
import copy
import argparse
import insightface
import onnxruntime
import numpy as np
from PIL import Image
from typing import List, Union, Dict, Set, Tuple
import glob
from pathlib import Path
import torch
import urllib.request
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
import io
import tempfile
import shutil
from typing import List, Optional
import uvicorn
from pydantic import BaseModel, HttpUrl
import requests



def getFaceSwapModel(model_path: str):
    model = insightface.model_zoo.get_model(model_path)
    return model


def getFaceAnalyser(model_path: str, providers,
                    det_size=(320, 320)):
    face_analyser = insightface.app.FaceAnalysis(
        name="buffalo_l", 
        providers=providers
    )
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    return face_analyser


def get_one_face(face_analyser,
                 frame:np.ndarray):
    face = face_analyser.get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None

    
def get_many_faces(face_analyser,
                   frame:np.ndarray):
    """
    get faces from left to right by order
    """
    try:
        face = face_analyser.get(frame)
        if not face:  # Check if face list is empty
            print("No faces detected in image")
            return None
        return sorted(face, key=lambda x: x.bbox[0])
    except Exception as e:
        print(f"Error detecting faces: {str(e)}")
        return None


def swap_face(face_swapper,
              source_faces,
              target_faces,
              source_index,
              target_index,
              temp_frame):
    """
    paste source_face on target image
    """
    if source_faces is None or len(source_faces) <= source_index:
        raise Exception(f"Source face index {source_index} is invalid. Number of detected source faces: {len(source_faces) if source_faces else 0}")
    
    if target_faces is None or len(target_faces) <= target_index:
        raise Exception(f"Target face index {target_index} is invalid. Number of detected target faces: {len(target_faces) if target_faces else 0}")

    source_face = source_faces[source_index]
    target_face = target_faces[target_index]
    
    # Use inswapper model's built-in face swapping
    return face_swapper.get(temp_frame, target_face, source_face, paste_back=True)


def process(source_img: Union[Image.Image, List],
            target_img: Image.Image,
            source_indexes: str,
            target_indexes: str,
            model: str):
    # load machine default available providers
    providers = onnxruntime.get_available_providers()

    # load face_analyser
    face_analyser = getFaceAnalyser(model, providers)
    
    # load face_swapper
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
    face_swapper = getFaceSwapModel(model_path)
    
    # read target image
    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    
    # Add debug information
    for idx, img in enumerate(source_img):
        img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        faces = get_many_faces(face_analyser, img_array)
        print(f"Source image {idx}: detected {len(faces) if faces else 0} faces")

    target_faces = get_many_faces(face_analyser, target_img)
    print(f"Target image: detected {len(target_faces) if target_faces else 0} faces")

    num_target_faces = len(target_faces)
    num_source_images = len(source_img)

    if target_faces is not None:
        temp_frame = copy.deepcopy(target_img)
        if isinstance(source_img, list) and num_source_images == num_target_faces:
            print("Number of source images matches number of target faces")
            for i in range(num_target_faces):
                source_faces = get_many_faces(face_analyser, cv2.cvtColor(np.array(source_img[i]), cv2.COLOR_RGB2BGR))
                if source_faces is None or len(source_faces) == 0:
                    print(f"Warning: No face detected in source image {i}, skipping...")
                    continue
                    
                # Always use the first (or only) face from each source image
                source_index = 0
                target_index = i

                temp_frame = swap_face(
                    face_swapper,
                    source_faces,
                    target_faces,
                    source_index,
                    target_index,
                    temp_frame
                )
        elif num_source_images == 1:
            # detect source faces that will be replaced into the target image
            source_faces = get_many_faces(face_analyser, cv2.cvtColor(np.array(source_img[0]), cv2.COLOR_RGB2BGR))
            num_source_faces = len(source_faces)
            print(f"Source faces: {num_source_faces}")
            print(f"Target faces: {num_target_faces}")

            if source_faces is None:
                raise Exception("No source faces found!")

            if target_indexes == "-1":
                if num_source_faces == 1:
                    print("Replacing all faces in target image with the same face from the source image")
                    num_iterations = num_target_faces
                elif num_source_faces < num_target_faces:
                    print("There are less faces in the source image than the target image, replacing as many as we can")
                    num_iterations = num_source_faces
                elif num_target_faces < num_source_faces:
                    print("There are less faces in the target image than the source image, replacing as many as we can")
                    num_iterations = num_target_faces
                else:
                    print("Replacing all faces in the target image with the faces from the source image")
                    num_iterations = num_target_faces

                for i in range(num_iterations):
                    source_index = 0 if num_source_faces == 1 else i
                    target_index = i

                    temp_frame = swap_face(
                        face_swapper,
                        source_faces,
                        target_faces,
                        source_index,
                        target_index,
                        temp_frame
                    )
            else:
                print("Replacing specific face(s) in the target image with specific face(s) from the source image")

                if source_indexes == "-1":
                    source_indexes = ','.join(map(lambda x: str(x), range(num_source_faces)))

                if target_indexes == "-1":
                    target_indexes = ','.join(map(lambda x: str(x), range(num_target_faces)))

                source_indexes = source_indexes.split(',')
                target_indexes = target_indexes.split(',')
                num_source_faces_to_swap = len(source_indexes)
                num_target_faces_to_swap = len(target_indexes)

                if num_source_faces_to_swap > num_source_faces:
                    raise Exception("Number of source indexes is greater than the number of faces in the source image")

                if num_target_faces_to_swap > num_target_faces:
                    raise Exception("Number of target indexes is greater than the number of faces in the target image")

                if num_source_faces_to_swap > num_target_faces_to_swap:
                    num_iterations = num_source_faces_to_swap
                else:
                    num_iterations = num_target_faces_to_swap

                if num_source_faces_to_swap == num_target_faces_to_swap:
                    for index in range(num_iterations):
                        source_index = int(source_indexes[index])
                        target_index = int(target_indexes[index])

                        if source_index > num_source_faces-1:
                            raise ValueError(f"Source index {source_index} is higher than the number of faces in the source image")

                        if target_index > num_target_faces-1:
                            raise ValueError(f"Target index {target_index} is higher than the number of faces in the target image")

                        temp_frame = swap_face(
                            face_swapper,
                            source_faces,
                            target_faces,
                            source_index,
                            target_index,
                            temp_frame
                        )
        else:
            raise Exception("Unsupported face configuration")
        result = temp_frame
    else:
        print("No target faces found!")
    
    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return result_image


def parse_args():
    parser = argparse.ArgumentParser(description="Face swap.")
    parser.add_argument("--source_img", type=str, required=True, help="The path of source image, it can be multiple images, dir;dir2;dir3.")
    parser.add_argument("--target_img", type=str, required=True, help="The path of target image.")
    parser.add_argument("--output_img", type=str, required=False, default="result.png", help="The path and filename of output image.")
    parser.add_argument("--source_indexes", type=str, required=False, default="-1", help="Comma separated list of the face indexes to use (left to right) in the source image, starting at 0 (-1 uses all faces in the source image")
    parser.add_argument("--target_indexes", type=str, required=False, default="-1", help="Comma separated list of the face indexes to swap (left to right) in the target image, starting at 0 (-1 swaps all faces in the target image")
    parser.add_argument("--face_restore", action="store_true", help="The flag for face restoration.")
    parser.add_argument("--background_enhance", action="store_true", help="The flag for background enhancement.")
    parser.add_argument("--face_upsample", action="store_true", help="The flag for face upsample.")
    parser.add_argument("--upscale", type=int, default=1, help="The upscale value, up to 4.")
    parser.add_argument("--codeformer_fidelity", type=float, default=0.5, help="The codeformer fidelity.")
    args = parser.parse_args()
    return args


def get_images_from_folder(folder_path):
    """Get all images from a folder"""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    images = []
    for ext in extensions:
        images.extend(glob.glob(str(Path(folder_path) / ext)))
    return images


def download_model():
    """Download the inswapper model if it doesn't exist"""
    model_path = "./checkpoints/inswapper_128.onnx"
    if not os.path.exists(model_path):
        print("Downloading inswapper model...")
        os.makedirs("./checkpoints", exist_ok=True)
        url = "https://www.dropbox.com/scl/fi/h8rwajkgfrfw72w5yfbct/inswapper_128.onnx?rlkey=avqyrpfmxfxcmz8xsipsgpmg9&st=wy47uc4t&dl=1"
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded successfully")


def download_and_extract_buffalo_model():
    """Download buffalo_l model files"""
    import urllib.request
    import os

    buffalo_path = "checkpoints/models/buffalo_l"
    os.makedirs(buffalo_path, exist_ok=True)
    
    model_urls = {
        'w600k_r50.onnx': "https://www.dropbox.com/scl/fi/a1dthaiglolxqf51gp6jb/w600k_r50.onnx?rlkey=mtafser7afgcqa7218g5s3tn3&st=l8p8cfha&dl=1",
        'genderage.onnx': "https://www.dropbox.com/scl/fi/5sehilvdn13y93091trs4/genderage.onnx?rlkey=gpocnlmys0ixtkkri8dnwwsvz&st=5dhrltwv&dl=1",
        'det_10g.onnx': "https://www.dropbox.com/scl/fi/gv67fx8vtc7phg7l7h1s5/det_10g.onnx?rlkey=wlgqbkdtrzfcg506vxpvg6n8j&st=lvvn3tdl&dl=1",
        '2d106det.onnx': "https://www.dropbox.com/scl/fi/ly3kgdf8hg2r7eqfab4e4/2d106det.onnx?rlkey=h43adi8jnfv0he90yaatebc4k&st=58hpwfil&dl=1",
        '1k3d68.onnx': "https://www.dropbox.com/scl/fi/sj5v97t4s7s3pjmnpn97j/1k3d68.onnx?rlkey=1gnmdn93y1djl4zjomucgaeb6&st=twgha4t4&dl=1"
    }

    for filename, url in model_urls.items():
        file_path = os.path.join(buffalo_path, filename)
        if not os.path.exists(file_path):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, file_path)
            print(f"{filename} downloaded successfully")

    print("All buffalo_l model files downloaded successfully")


def check_models_exist():
    """Check if all required model files exist"""
    buffalo_path = "checkpoints/models/buffalo_l"
    inswapper_path = "./checkpoints/inswapper_128.onnx"
    
    # Check both paths
    if not os.path.exists(inswapper_path):
        return False
        
    # Check buffalo model files
    required_files = [
        'w600k_r50.onnx',
        'genderage.onnx',
        'det_10g.onnx',
        '2d106det.onnx',
        '1k3d68.onnx'
    ]
    
    for filename in required_files:
        if not os.path.exists(os.path.join(buffalo_path, filename)):
            return False
            
    return True


def process_all_images():
    """Process all images from source and target folders"""
    # Check if models exist first
    if not check_models_exist():
        print("Downloading required models...")
        download_model()  # for inswapper
        download_and_extract_buffalo_model()  # for buffalo_l
    else:
        print("All required models found.")

    # Setup paths
    data_dir = Path("data")
    source_dir = data_dir / "source"
    target_dir = data_dir / "target"
    faceswap_dir = data_dir / "faceswap"  # New directory for faceswap results
    output_dir = data_dir / "output"

    # Create output directories if they don't exist
    faceswap_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all images
    source_images = get_images_from_folder(source_dir)
    target_images = get_images_from_folder(target_dir)

    if not source_images:
        raise Exception(f"No source images found in {source_dir}")
    if not target_images:
        raise Exception(f"No target images found in {target_dir}")

    print(f"Found {len(source_images)} source images and {len(target_images)} target images")

    # Process each target image with each source image
    for target_path in target_images:
        target_name = Path(target_path).stem
        target_img = Image.open(target_path)

        for source_path in source_images:
            source_name = Path(source_path).stem
            source_img = [Image.open(source_path)]  # Wrap in list as process() expects a list

            # Create output filenames
            faceswap_name = f"{source_name}_to_{target_name}.png"
            faceswap_path = str(faceswap_dir / faceswap_name)
            final_output_path = str(output_dir / faceswap_name)

            print(f"\nProcessing: {source_name} -> {target_name}")
            
            try:
                # Step 1: Face Swap
                print("Performing face swap...")
                model = "./checkpoints/inswapper_128.onnx"
                result_image = process(source_img, target_img, "-1", "-1", model)
                
                # Save face swap result
                result_image.save(faceswap_path)
                print(f'Face swap result saved: {faceswap_path}')

                # Step 2: Face Restoration
                print("Applying face restoration...")
                from restoration import face_restoration, set_realesrgan, check_ckpts, ARCH_REGISTRY
                
                # make sure the ckpts downloaded successfully
                check_ckpts()
                
                # https://huggingface.co/spaces/sczhou/CodeFormer
                upsampler = set_realesrgan()
                device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

                codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
                    dim_embd=512,
                    codebook_size=1024,
                    n_head=8,
                    n_layers=9,
                    connect_list=["32", "64", "128", "256"],
                ).to(device)
                
                ckpt_path = "CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth"
                checkpoint = torch.load(ckpt_path, weights_only=True)["params_ema"]
                codeformer_net.load_state_dict(checkpoint)
                codeformer_net.eval()
                
                # Convert PIL Image to cv2 format for restoration
                result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
                restored_image = face_restoration(
                    result_image,
                    background_enhance=True,
                    face_upsample=True,
                    upscale=2,
                    codeformer_fidelity=0.7,
                    upsampler=upsampler,
                    codeformer_net=codeformer_net,
                    device=device
                )
                
                # Save final restored result
                final_result = Image.fromarray(restored_image)
                final_result.save(final_output_path)
                print(f'Final restored result saved: {final_output_path}')
                
            except Exception as e:
                print(f"Error processing {source_name} -> {target_name}: {str(e)}")
                continue

# Create FastAPI app instance
app = FastAPI(
    title="Face Swapper API",
    description="API for face swapping using insightface",
    version="1.0.0"
)

class SwapRequest(BaseModel):
    source_url: str
    target_url: str
    source_indexes: str = "-1"
    target_indexes: str = "-1"

class RestoreRequest(BaseModel):
    source_url: str
    target_url: str
    source_indexes: str = "-1"
    target_indexes: str = "-1"
    background_enhance: bool = True
    face_upsample: bool = True
    upscale: int = 2
    codeformer_fidelity: float = 0.7

async def download_image_from_url(url: str) -> str:
    """Download image from URL and save to temporary file"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file extension from content-type
        content_type = response.headers.get('content-type', '')
        ext = '.jpg' if 'jpeg' in content_type else '.png' if 'png' in content_type else '.jpg'
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            return tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")

def swap_process(source_img: List[Image.Image], 
                target_img: Image.Image,
                source_indexes: str,
                target_indexes: str,
                model_path: str):
    """Helper function for face swapping process"""
    # load machine default available providers
    providers = onnxruntime.get_available_providers()

    # load face_analyser
    face_analyser = getFaceAnalyser(model_path, providers)
    
    # load face_swapper
    face_swapper = getFaceSwapModel(model_path)
    
    # read target image
    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    
    # Add debug information
    for idx, img in enumerate(source_img):
        img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        faces = get_many_faces(face_analyser, img_array)
        print(f"Source image {idx}: detected {len(faces) if faces else 0} faces")

    target_faces = get_many_faces(face_analyser, target_img)
    print(f"Target image: detected {len(target_faces) if target_faces else 0} faces")

    if target_faces is None:
        raise HTTPException(status_code=400, detail="No faces detected in target image")

    temp_frame = copy.deepcopy(target_img)
    source_faces = get_many_faces(face_analyser, cv2.cvtColor(np.array(source_img[0]), cv2.COLOR_RGB2BGR))
    
    if source_faces is None:
        raise HTTPException(status_code=400, detail="No faces detected in source image")

    temp_frame = swap_face(
        face_swapper,
        source_faces,
        target_faces,
        0,  # Using first face from source
        0,  # Using first face from target
        temp_frame
    )
    
    return Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))

@app.post("/swap-face/")
async def swap_face_endpoint(request: SwapRequest):
    """
    Endpoint to perform face swapping between source and target images.
    Both images must be provided as URLs.
    """
    try:
        # First check if models exist and download if needed
        if not check_models_exist():
            print("Downloading required models...")
            download_model()
            download_and_extract_buffalo_model()

        # Download images from URLs
        source_path = await download_image_from_url(request.source_url)
        target_path = await download_image_from_url(request.target_url)

        # Open images
        source_img = [Image.open(source_path)]
        target_img = Image.open(target_path)

        # Process the face swap
        model_path = "./checkpoints/inswapper_128.onnx"
        result_image = swap_process(
            source_img, 
            target_img, 
            request.source_indexes, 
            request.target_indexes, 
            model_path
        )

        # Save result to temporary file
        output_path = tempfile.mktemp(suffix='.png')
        result_image.save(output_path)

        # Clean up temporary files
        os.unlink(source_path)
        os.unlink(target_path)

        # Return the result image
        return FileResponse(output_path, media_type="image/png", filename="result.png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/swap-face-restore/")
async def swap_face_restore_endpoint(request: RestoreRequest):
    """
    Endpoint to perform face swapping with face restoration.
    Both images must be provided as URLs.
    """
    try:
        # Check models
        if not check_models_exist():
            print("Downloading required models...")
            download_model()
            download_and_extract_buffalo_model()

        # Download images from URLs
        source_path = await download_image_from_url(request.source_url)
        target_path = await download_image_from_url(request.target_url)

        # Open images
        source_img = [Image.open(source_path)]
        target_img = Image.open(target_path)

        # Process face swap
        model_path = "./checkpoints/inswapper_128.onnx"
        result_image = swap_process(
            source_img, 
            target_img, 
            request.source_indexes, 
            request.target_indexes, 
            model_path
        )

        # Convert to cv2 format for restoration
        result_image_cv2 = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)

        # Perform face restoration
        from restoration import face_restoration, set_realesrgan, check_ckpts, ARCH_REGISTRY
        
        print(f"Starting restoration with parameters:")
        print(f"- Background enhance: {request.background_enhance}")
        print(f"- Face upsample: {request.face_upsample}")
        print(f"- Upscale: {request.upscale}")
        print(f"- CodeFormer fidelity: {request.codeformer_fidelity}")

        check_ckpts()
        upsampler = set_realesrgan()
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

        codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(device)
        
        ckpt_path = "CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth"
        checkpoint = torch.load(ckpt_path, weights_only=True)["params_ema"]
        codeformer_net.load_state_dict(checkpoint)
        codeformer_net.eval()

        # Apply restoration with user-specified parameters
        restored_image = face_restoration(
            result_image_cv2,
            background_enhance=request.background_enhance,
            face_upsample=request.face_upsample,
            upscale=request.upscale,
            codeformer_fidelity=request.codeformer_fidelity,
            upsampler=upsampler,
            codeformer_net=codeformer_net,
            device=device
        )

        # Save result to temporary file
        output_path = tempfile.mktemp(suffix='.png')
        final_result = Image.fromarray(restored_image)
        final_result.save(output_path)

        # Clean up temporary files
        os.unlink(source_path)
        os.unlink(target_path)

        # Return the result image
        return FileResponse(
            output_path, 
            media_type="image/png", 
            filename=f"result_restored_f{request.codeformer_fidelity}_x{request.upscale}.png"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add this at the bottom of the file
if __name__ == "__main__":
    uvicorn.run("swapper:app", host="0.0.0.0", port=8000, reload=True)