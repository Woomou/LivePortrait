import numpy as np
from typing import Generator
from ..live_portrait_pipeline import LivePortraitPipeline
from ..config.argument_config import ArgumentConfig
from ..config.crop_config import CropConfig
from ..utils.io import load_image_rgb, load_video, resize_to_limit, load
from ..utils.helper import is_image, is_video, is_template, dct2device
from ..utils.camera import get_rotation_matrix  # 新增导入
import cv2
import os.path as osp

class FrameGenerator:
    """
    逐帧生成器对象：每次调用frame_iter()可一帧一帧地生成推理结果。
    支持循环播放（loop）。
    """
    def __init__(self, pipeline: LivePortraitPipeline, args: ArgumentConfig):
        self.pipeline = pipeline
        self.args = args
        self.inf_cfg = pipeline.live_portrait_wrapper.inference_cfg
        self.device = pipeline.live_portrait_wrapper.device
        self.crop_cfg = pipeline.cropper.crop_cfg

    def frame_iter(self, loop: bool = True) -> Generator[np.ndarray, None, None]:
        """
        逐帧生成推理结果，每次yield一帧（np.ndarray，HWC，uint8）。
        loop=True时，自动循环播放。
        """
        args = self.args
        inf_cfg = self.inf_cfg
        device = self.device
        crop_cfg = self.crop_cfg

        # 载入source
        if is_image(args.source):
            img_rgb = load_image_rgb(args.source)
            img_rgb = resize_to_limit(img_rgb, inf_cfg.source_max_dim, inf_cfg.source_division)
            source_rgb_lst = [img_rgb]
            flag_is_source_video = False
        elif is_video(args.source):
            source_rgb_lst = load_video(args.source)
            source_rgb_lst = [resize_to_limit(img, inf_cfg.source_max_dim, inf_cfg.source_division) for img in source_rgb_lst]
            flag_is_source_video = True
        else:
            raise Exception(f"Unknown source format: {args.source}")

        # 载入driving
        flag_load_from_template = is_template(args.driving)
        if flag_load_from_template:
            driving_template_dct = load(args.driving)
            n_frames = driving_template_dct['n_frames']
        elif osp.exists(args.driving) and is_video(args.driving):
            driving_rgb_lst = load_video(args.driving)
            n_frames = len(driving_rgb_lst)
        else:
            raise Exception(f"Unknown or unsupported driving format: {args.driving}")

        # 只支持source为图片，driving为视频/模板的常见用法
        # 可根据需要扩展
        # 预处理source
        img_crop_256x256 = cv2.resize(source_rgb_lst[0], (256, 256))
        # 新增：全流程半精度，转为float16
        img_crop_256x256 = img_crop_256x256.astype(np.float16)
        I_s = self.pipeline.live_portrait_wrapper.prepare_source(img_crop_256x256)
        if hasattr(I_s, 'half'):
            I_s = I_s.half()
        x_s_info = self.pipeline.live_portrait_wrapper.get_kp_info(I_s)
        x_c_s = x_s_info['kp']
        R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])  # 修正调用
        f_s = self.pipeline.live_portrait_wrapper.extract_feature_3d(I_s)
        if hasattr(f_s, 'half'):
            f_s = f_s.half()
        x_s = self.pipeline.live_portrait_wrapper.transform_keypoint(x_s_info)

        idx = 0
        while True:
            if flag_load_from_template:
                i = idx % n_frames
                x_d_i_info = driving_template_dct['motion'][i]
                x_d_i_info = dct2device(x_d_i_info, device)
                R_d_i = x_d_i_info['R'] if 'R' in x_d_i_info.keys() else x_d_i_info['R_d']
                delta_new = x_d_i_info['exp']
                t_new = x_d_i_info['t']
                t_new[..., 2].fill_(0)
                scale_new = x_s_info['scale']
                x_d_i = scale_new * (x_c_s @ R_d_i + delta_new) + t_new
            else:
                i = idx % n_frames
                driving_img = cv2.resize(driving_rgb_lst[i], (256, 256))
                # 新增：全流程半精度，转为float16
                driving_img = driving_img.astype(np.float16)
                I_d = self.pipeline.live_portrait_wrapper.prepare_source(driving_img)
                if hasattr(I_d, 'half'):
                    I_d = I_d.half()
                x_d_i_info = self.pipeline.live_portrait_wrapper.get_kp_info(I_d)
                R_d_i = get_rotation_matrix(x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])  # 修正调用
                delta_new = x_d_i_info['exp']
                t_new = x_d_i_info['t']
                t_new[..., 2].fill_(0)
                scale_new = x_s_info['scale']
                x_d_i = scale_new * (x_c_s @ R_d_i + delta_new) + t_new
            out = self.pipeline.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i)
            # 新增：输出转为float16再parse_output
            if hasattr(out['out'], 'half'):
                out_tensor = out['out'].half()
            else:
                out_tensor = out['out']
            I_p_i = self.pipeline.live_portrait_wrapper.parse_output(out_tensor)[0]
            yield I_p_i
            idx += 1
            if not loop and idx >= n_frames:
                break
