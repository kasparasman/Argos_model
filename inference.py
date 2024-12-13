from glob import glob
import shutil
import torch
from time import  strftime
import os, sys, time
from argparse import ArgumentParser
import platform
from contextlib import contextmanager
from collections import defaultdict

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.facerender.pirender_animate import AnimateFromCoeff_PIRender
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
#temporary
def download_video(file_path):
    """
    Automatically copy the video to the /content/Argos_model/results folder.
    """
    target_folder = "/content/Argos_model/results"
    os.makedirs(target_folder, exist_ok=True)  # Ensure the folder exists
    if os.path.exists(file_path):
        local_filename = os.path.basename(file_path)
        target_path = os.path.join(target_folder, local_filename)
        print(f"Copying video to: {target_path}")
        try:
            shutil.copy(file_path, target_path)
            print(f"Video successfully copied to {target_path}")
        except Exception as e:
            print(f"Error copying video: {e}")
    else:
        print(f"Video file not found at {file_path}")

class TimingStats:
    _timings = defaultdict(list)
    
    @classmethod
    @contextmanager
    def timer(cls, name):
        start = time.time()
        yield
        duration = time.time() - start
        cls._timings[name].append(duration)
        print(f"⏱️ {name}: {duration:.3f}s")
    
    @classmethod
    def print_summary(cls):
        print("\n=== Timing Summary ===")
        for name, times in cls._timings.items():
            avg_time = sum(times) / len(times)
            total_time = sum(times)
            print(f"{name}:")
            print(f"  Average: {avg_time:.3f}s")
            print(f"  Total: {total_time:.3f}s")
            print(f"  Calls: {len(times)}")


def main(args):
    pic_path = args.source_image
    audio_path = args.driven_audio
    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    pose_style = args.pose_style
    device = args.device
    batch_size = args.batch_size
    input_yaw_list = args.input_yaw
    input_pitch_list = args.input_pitch
    input_roll_list = args.input_roll
    ref_eyeblink = args.ref_eyeblink
    ref_pose = args.ref_pose
    
    with TimingStats.timer("Full Pipeline"):
        with TimingStats.timer("Initialization"):
            current_root_path = os.path.split(sys.argv[0])[0]
            sadtalker_paths = init_path(args.checkpoint_dir, os.path.join(current_root_path, 'src/config'), 
                                      args.size, args.old_version, args.preprocess)
            
            print("Loading models...")
            preprocess_model = CropAndExtract(sadtalker_paths, device)
            audio_to_coeff = Audio2Coeff(sadtalker_paths, device)
            
            if args.facerender == 'facevid2vid':
                animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)
            elif args.facerender == 'pirender':
                animate_from_coeff = AnimateFromCoeff_PIRender(sadtalker_paths, device)
            else:
                raise(RuntimeError('Unknown model: {}'.format(args.facerender)))

        first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)
        
        # Preprocess - 3DMM Extraction
        print('3DMM Extraction for source image...')
        with TimingStats.timer("3DMM Extraction"):
            first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
                pic_path, first_frame_dir, args.preprocess,
                source_image_flag=True, pic_size=args.size
            )
            
            if first_coeff_path is None:
                print("Can't get the coeffs of the input")
                return

        # Audio to coefficients
        with TimingStats.timer("Audio Processing"):
            batch = get_data(first_coeff_path, audio_path, device, 
                           ref_eyeblink, still=args.still)
            
        with TimingStats.timer("Coeff Generation"):
            coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose)

        # Face rendering
        with TimingStats.timer("Data Preparation"):
            data = get_facerender_data(
                coeff_path, crop_pic_path, first_coeff_path, audio_path,
                batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                expression_scale=args.expression_scale, still_mode=args.still,
                preprocess=args.preprocess, size=args.size
            )

        with TimingStats.timer("Face Rendering"):
            result = animate_from_coeff.generate(
                data, save_dir, pic_path, crop_info,
                enhancer=args.enhancer,
                preprocess=args.preprocess, img_size=args.size
            )
            print(f"Video generation completed. Result path: {result}")
            if os.path.exists(result):
                print(f"Video file {result} exists.")
                download_video(result)  # Automatically download the generated video
            else:
                print(f"Video file {result} does not exist.")
            print(os.listdir(save_dir))
            print("Checking if the video file exists...")
            print(f"Save directory: {save_dir}")
            if os.path.exists(result):
                print(f"Video file {result} exists.")
            else:
                print(f"Video file {result} does not exist.")

    TimingStats.print_summary()

    if not args.verbose:
        shutil.rmtree(save_dir)

    
if __name__ == '__main__':

    parser = ArgumentParser()  
    parser.add_argument("--driven_audio", default='./examples/driven_audio/bus_chinese.wav', help="path to driven audio")
    parser.add_argument("--source_image", default='./examples/source_image/full_body_1.png', help="path to source image")
    parser.add_argument("--ref_eyeblink", default=None, help="path to reference video providing eye blinking")
    parser.add_argument("--ref_pose", default=None, help="path to reference video providing pose")
    parser.add_argument("--checkpoint_dir", default='./checkpoints', help="path to output")
    parser.add_argument("--result_dir", default='./results', help="path to output")
    parser.add_argument("--pose_style", type=int, default=0,  help="input pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=2,  help="the batch size of facerender")
    parser.add_argument("--size", type=int, default=256,  help="the image size of the facerender")
    parser.add_argument("--expression_scale", type=float, default=1.,  help="the batch size of facerender")
    parser.add_argument('--input_yaw', nargs='+', type=int, default=None, help="the input yaw degree of the user ")
    parser.add_argument('--input_pitch', nargs='+', type=int, default=None, help="the input pitch degree of the user")
    parser.add_argument('--input_roll', nargs='+', type=int, default=None, help="the input roll degree of the user")
    parser.add_argument('--enhancer',  type=str, default=None, help="Face enhancer, [gfpgan, RestoreFormer]")
    parser.add_argument('--background_enhancer',  type=str, default=None, help="background enhancer, [realesrgan]")
    parser.add_argument("--cpu", dest="cpu", action="store_true") 
    parser.add_argument("--face3dvis", action="store_true", help="generate 3d face and 3d landmarks") 
    parser.add_argument("--still", action="store_true", help="can crop back to the original videos for the full body aniamtion") 
    parser.add_argument("--preprocess", default='crop', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'], help="how to preprocess the images" ) 
    parser.add_argument("--verbose",action="store_true", help="saving the intermedia output or not" ) 
    parser.add_argument("--old_version",action="store_true", help="use the pth other than safetensor version" ) 
    parser.add_argument("--facerender", default='pirender', choices=['pirender', 'facevid2vid'] ) 
    

    # net structure and parameters
    parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='useless')
    parser.add_argument('--init_path', type=str, default=None, help='Useless')
    parser.add_argument('--use_last_fc',default=False, help='zero initialize the last fc')
    parser.add_argument('--bfm_folder', type=str, default='./checkpoints/BFM_Fitting/')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

    # default renderer parameters
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)

    args = parser.parse_args()

    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda"
    elif platform.system() == 'Darwin' and args.facerender == 'pirender': # macos 
        args.device = "mps"
    else:
        args.device = "cpu"

    main(args)
