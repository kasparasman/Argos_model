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


import imageio
import torch

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
        opt = Config(sadtalker_path['pirender_yaml_path'], None, is_train=False)
        opt.device = device
        self.net_G_ema = FaceGenerator(**opt.gen.param).to(opt.device)
        checkpoint_path = sadtalker_path['pirender_checkpoint']
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.net_G_ema.load_state_dict(checkpoint['net_G_ema'], strict=False)
        print('load [net_G] and [net_G_ema] from {}'.format(checkpoint_path))
        self.net_G = self.net_G_ema.eval()
        self.device = device
    

    def generate(self, x, video_save_dir, pic_path, crop_info, enhancer=None, preprocess='crop', img_size=256):
        source_image = x['source_image'].type(torch.FloatTensor)
        source_semantics = x['source_semantics'].type(torch.FloatTensor)
        target_semantics = x['target_semantics_list'].type(torch.FloatTensor)
        
        source_image = source_image.to(self.device)
        source_semantics = source_semantics.to(self.device)
        target_semantics = target_semantics.to(self.device)
        frame_num = x['frame_num']
        
        BATCH_SIZE = 4  # Start with a small batch size
        
        with torch.no_grad():
            predictions_video = []
            for i in tqdm(range(0, target_semantics.shape[1], BATCH_SIZE), 'FaceRender:'):
                batch_end = min(i + BATCH_SIZE, target_semantics.shape[1])
                current_batch_size = batch_end - i
                print(f"Processing batch {i // BATCH_SIZE + 1}, frames {i} to {batch_end}")
                # Handle both source image and semantics dimensions
                current_target_semantics = target_semantics[:, i:batch_end].squeeze(0)  # Remove the first dimension
                current_source_image = source_image[0].unsqueeze(0).expand(current_batch_size, -1, -1, -1)
                print(f"Before reshape: {current_target_semantics.shape}")
                try:
                    current_target_semantics = current_target_semantics.permute(1, 2, 0, 3)
                    current_target_semantics = current_target_semantics.reshape(
                        current_target_semantics.size(0),  # Batch size
                        current_target_semantics.size(1),  # Channels
                        -1  # Combine other dims into one
                    )
                    print(f"After reshape: {current_target_semantics.shape}")
                except Exception as e:
                    print(f"Permute error: {e}")
                batch_predictions = self.net_G(current_source_image, current_target_semantics)['fake_image']
                predictions_video.append(batch_predictions)
                print(f"Batch predictions shape: {batch_predictions.shape}")
            # Concatenate all batches
            predictions_video = torch.cat(predictions_video, dim=0)

        video = []
        for idx in range(len(predictions_video)):
            image = predictions_video[idx]
            image = np.transpose(image.data.cpu().numpy(), [1, 2, 0]).astype(np.float32)
            video.append(image)
        result = img_as_ubyte(video)
        print(f"Processed {len(result)} frames for video generation")

        ### the generated video is 256x256, so we keep the aspect ratio, 
        original_size = crop_info[0]
        if original_size:
            result = [ cv2.resize(result_i,(img_size, int(img_size * original_size[1]/original_size[0]) )) for result_i in result ]
        
        video_name = x['video_name']  + '.mp4'
        path = os.path.join(video_save_dir, 'temp_'+video_name)
        print(f"Saving temporary video to: {path}")

        imageio.mimsave(path, result,  fps=float(25))

        av_path = os.path.join(video_save_dir, video_name)
        return_path = av_path 
        
        audio_path =  x['audio_path'] 
        audio_name = os.path.splitext(os.path.split(audio_path)[-1])[0]
        new_audio_path = os.path.join(video_save_dir, audio_name+'.wav')
        start_time = 0
        sound = AudioSegment.from_file(audio_path)
        frames = frame_num 
        end_time = start_time + frames*1/25*1000
        word1=sound.set_frame_rate(16000)
        word = word1[start_time:end_time]
        word.export(new_audio_path, format="wav")
        print(f"Final video will be saved at: {av_path}")

        save_video_with_watermark(path, new_audio_path, av_path, watermark= False)
        print(f'The generated video is named {video_save_dir}/{video_name}') 

        if 'full' in preprocess.lower():
            video_name_full = x['video_name']  + '_full.mp4'
            full_video_path = os.path.join(video_save_dir, video_name_full)
            return_path = full_video_path
            paste_pic(path, pic_path, crop_info, new_audio_path, full_video_path, extended_crop= True if 'ext' in preprocess.lower() else False)
            print(f'The generated video is named {video_save_dir}/{video_name_full}') 
        else:
            full_video_path = av_path 

        if enhancer:
            video_name_enhancer = x['video_name']  + '_enhanced.mp4'
            enhanced_path = os.path.join(video_save_dir, 'temp_'+video_name_enhancer)
            av_path_enhancer = os.path.join(video_save_dir, video_name_enhancer) 
            return_path = av_path_enhancer

            try:
                enhanced_images_gen_with_len = enhancer_generator_with_len(full_video_path, method=enhancer)
                imageio.mimsave(enhanced_path, enhanced_images_gen_with_len, fps=float(25))
            except:
                enhanced_images_gen_with_len = enhancer_list(full_video_path, method=enhancer)
                imageio.mimsave(enhanced_path, enhanced_images_gen_with_len, fps=float(25))
            
            save_video_with_watermark(enhanced_path, new_audio_path, av_path_enhancer, watermark= False)
            print(f'The generated video is named {video_save_dir}/{video_name_enhancer}')
            os.remove(enhanced_path)
        print(f"Contents of {video_save_dir}: {os.listdir(video_save_dir)}")
        if os.path.exists(av_path):
            print(f"File exists: {av_path}")
        else:
            print(f"File not found: {av_path}")
        os.remove(path)
        os.remove(new_audio_path)

        return return_path
