import os
import cv2
from tqdm import tqdm
import yaml
import numpy as np
import warnings
from skimage import img_as_ubyte
import safetensors
import safetensors.torch
warnings.filterwarnings('ignore')
import time 

import imageio
import torch
# Add at the top with other imports
from src.utils.profiling_utils import SadTalkerProfiler, profile_generator
from src.facerender.pirender.config import Config
from src.facerender.pirender.face_model import FaceGenerator

from pydub import AudioSegment
from src.utils.face_enhancer import enhancer_generator_with_len, enhancer_list
from src.utils.paste_pic import paste_pic
from src.utils.videoio import save_video_with_watermark

try:
    import webui  # in webui
    in_webui = True
except:
    in_webui = False

class AnimateFromCoeff_PIRender():
    def __init__(self, sadtalker_path, device):
        self.profiler = SadTalkerProfiler()
        early_init_start = time.perf_counter()  # Add this at the very start
        
        # Track imports
        import_start = time.perf_counter()
        import_time = time.perf_counter() - import_start
        self.profiler.logger.debug(f"Early imports time: {import_time:.3f}s")
        
        # Track CUDA setup
        cuda_start = time.perf_counter()
        torch.backends.cudnn.benchmark = True
        cuda_time = time.perf_counter() - cuda_start
        self.profiler.logger.debug(f"CUDA setup time: {cuda_time:.3f}s")
        # Initialize profiler
        self.profiler.start_event("total_init")
        
        # Original initialization code
        self.profiler.start_event("config_loading")
        opt = Config(sadtalker_path['pirender_yaml_path'], None, is_train=False)
        opt.device = device
        self.profiler.end_event("config_loading")

        # Initialize model and convert to half precision
        self.profiler.start_event("model_creation")
        self.net_G_ema = FaceGenerator(**opt.gen.param).to(opt.device).half()
        self.profiler.end_event("model_creation")

        self.profiler.start_event("checkpoint_loading")
        checkpoint_path = sadtalker_path['pirender_checkpoint']
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        # Convert checkpoint weights to half precision
        for k, v in checkpoint['net_G_ema'].items():
            checkpoint['net_G_ema'][k] = v.half()

        self.net_G_ema.load_state_dict(checkpoint['net_G_ema'], strict=False)
        print('load [net_G] and [net_G_ema] from {}'.format(checkpoint_path))
        self.net_G = self.net_G_ema.eval()
        self.profiler.end_event("checkpoint_loading")
        
        self.profiler.start_event("tensorrt_conversion")
        print("Converting model to TensorRT...")
        from torch2trt import torch2trt

        # Create sample input for TensorRT conversion
        sample_source = torch.randn(1, 3, 256, 256).to(device).half()
        sample_semantics = torch.randn(1, 73, 54).to(device).half()

        # Convert to TensorRT
        try:
            self.trt_model = torch2trt(
                self.net_G,
                [sample_source, sample_semantics],
                fp16_mode=True,
                max_batch_size=32,
                max_workspace_size=1 << 30
            )
            print("TensorRT conversion successful")
        except Exception as e:
            print(f"TensorRT conversion failed: {e}")
            print("Falling back to regular model")
            self.trt_model = None
        self.profiler.end_event("tensorrt_conversion")

        self.device = device
        self.profiler.end_event("total_init")
    @profile_generator
    def generate(self, x, video_save_dir, pic_path, crop_info, enhancer=None, preprocess='crop', img_size=256):
        self.profiler.start_event("generate_total")
        
        # Data preparation
        self.profiler.start_event("data_preparation")
        source_image = x['source_image'].type(torch.FloatTensor)
        source_semantics = x['source_semantics'].type(torch.FloatTensor)
        target_semantics = x['target_semantics_list'].type(torch.FloatTensor)
    
        source_image = source_image.to(self.device).half()
        source_semantics = source_semantics.to(self.device).half()
        target_semantics = target_semantics.to(self.device).half()
        frame_num = x['frame_num']
        total_frames = target_semantics.shape[1]
        self.profiler.end_event("data_preparation")
    
        BATCH_SIZE = 16
    
        with torch.no_grad():
            # Output tensor allocation
            self.profiler.start_event("output_allocation")
            predictions_video = torch.empty(
                (total_frames, 3, img_size, img_size),
                dtype=torch.float16,
                device=self.device
            )
            self.profiler.end_event("output_allocation")
    
            # Batch processing loop
            self.profiler.start_event("batch_processing")
            for i in tqdm(range(0, total_frames, BATCH_SIZE), 'FaceRender:'):
                self.profiler.start_event(f"batch_{i//BATCH_SIZE}")
                
                batch_end = min(i + BATCH_SIZE, total_frames)
                current_batch_size = batch_end - i
    
                # Batch preparation
                self.profiler.start_event(f"batch_{i//BATCH_SIZE}_prep")
                current_target_semantics = target_semantics[:, i:batch_end].squeeze(0)
                current_source_image = source_image[0].unsqueeze(0).expand(current_batch_size, -1, -1, -1)
    
                current_target_semantics = current_target_semantics.permute(1, 2, 0, 3)
                current_target_semantics = current_target_semantics.reshape(
                    current_target_semantics.size(0),
                    current_target_semantics.size(1),
                    -1
                )
                self.profiler.end_event(f"batch_{i//BATCH_SIZE}_prep")
    
                # Model inference
                self.profiler.start_event(f"batch_{i//BATCH_SIZE}_inference")
                if self.trt_model is not None:
                    batch_predictions = self.trt_model(current_source_image, current_target_semantics)['fake_image']
                else:
                    batch_predictions = self.net_G(current_source_image, current_target_semantics)['fake_image']
                predictions_video[i:batch_end] = batch_predictions
                self.profiler.end_event(f"batch_{i//BATCH_SIZE}_inference")
                
                self.profiler.end_event(f"batch_{i//BATCH_SIZE}")
            self.profiler.end_event("batch_processing")
    
            # Post-processing
            self.profiler.start_event("post_processing")
            predictions_cpu = predictions_video.float().cpu().numpy()
            video_frames = []
            for frame in predictions_cpu:
                frame = np.transpose(frame, [1, 2, 0]).astype(np.float32)
                video_frames.append(frame)
    
            result = img_as_ubyte(video_frames)
            self.profiler.end_event("post_processing")
    
            # Resizing if needed
            self.profiler.start_event("resizing")
            original_size = crop_info[0]
            if original_size:
                result = [cv2.resize(frame, (img_size, int(img_size * original_size[1]/original_size[0])))
                         for frame in result]
            self.profiler.end_event("resizing")
    
            # Video saving
            self.profiler.start_event("video_saving")
            video_name = x['video_name'] + '.mp4'
            temp_path = os.path.join(video_save_dir, 'temp_' + video_name)
            imageio.mimsave(temp_path, result, fps=float(25))
            self.profiler.end_event("video_saving")
    
            # Audio processing
            self.profiler.start_event("audio_processing")
            av_path = os.path.join(video_save_dir, video_name)
            return_path = av_path
    
            audio_path = x['audio_path']
            audio_name = os.path.splitext(os.path.split(audio_path)[-1])[0]
            new_audio_path = os.path.join(video_save_dir, audio_name + '.wav')
    
            sound = AudioSegment.from_file(audio_path)
            end_time = frame_num * (1000/25)
            word = sound.set_frame_rate(16000)[:int(end_time)]
            word.export(new_audio_path, format="wav")
            self.profiler.end_event("audio_processing")
    
            # Final video saving
            self.profiler.start_event("final_video_save")
            save_video_with_watermark(temp_path, new_audio_path, av_path, watermark=False)
            self.profiler.end_event("final_video_save")
    
            # Full video processing if needed
            if 'full' in preprocess.lower():
                self.profiler.start_event("full_video_processing")
                video_name_full = x['video_name'] + '_full.mp4'
                full_video_path = os.path.join(video_save_dir, video_name_full)
                return_path = full_video_path
                paste_pic(temp_path, pic_path, crop_info, new_audio_path, full_video_path,
                         extended_crop=True if 'ext' in preprocess.lower() else False)
                self.profiler.end_event("full_video_processing")
            else:
                full_video_path = av_path
    
            # Enhancement if needed
            if enhancer:
                self.profiler.start_event("enhancement")
                video_name_enhancer = x['video_name'] + '_enhanced.mp4'
                enhanced_path = os.path.join(video_save_dir, 'temp_' + video_name_enhancer)
                av_path_enhancer = os.path.join(video_save_dir, video_name_enhancer)
                return_path = av_path_enhancer
    
                try:
                    enhanced_images = enhancer_generator_with_len(full_video_path, method=enhancer)
                    imageio.mimsave(enhanced_path, enhanced_images, fps=float(25))
                except:
                    enhanced_images = enhancer_list(full_video_path, method=enhancer)
                    imageio.mimsave(enhanced_path, enhanced_images, fps=float(25))
    
                save_video_with_watermark(enhanced_path, new_audio_path, av_path_enhancer, watermark=False)
                os.remove(enhanced_path)
                self.profiler.end_event("enhancement")
    
            # Cleanup
            self.profiler.start_event("cleanup")
            os.remove(temp_path)
            os.remove(new_audio_path)
            self.profiler.end_event("cleanup")
    
        self.profiler.end_event("generate_total")
        return return_path