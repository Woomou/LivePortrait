#!/usr/bin/env python3
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib
import numpy as np
import cv2
import threading
import time
import os

# ===== 新增：导入FrameGenerator及依赖 =====
from src.utils.frame_generator import FrameGenerator
from src.live_portrait_pipeline import LivePortraitPipeline
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig

# 初始化GStreamer
Gst.init(None)

class SensorFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, frame_generator, width=256, height=256, fps=30):
        super(SensorFactory, self).__init__()
        self.width = width
        self.height = height
        self.fps = fps
        self.number_frames = 0
        self.duration = 1 / fps * Gst.SECOND  # duration of a frame in nanoseconds
        self.frame_iter = frame_generator.frame_iter()  # 这里传入FrameGenerator
        self.latest_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)  # 初始化帧
        self.lock = threading.Lock()
        self.running = True
        self.gen_frame_count = 0  # 新增帧计数器
        self.launch_string = (
            f'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME '
            f'caps=video/x-raw,format=RGB,width={width},height={height},framerate={fps}/1 '
            '! videoconvert ! openh264enc ! h264parse '
            '! rtph264pay name=pay0 pt=96 config-interval=1'
        )
        # 启动后台线程持续生成帧
        self.gen_thread = threading.Thread(target=self._frame_gen_loop, daemon=True)
        self.gen_thread.start()

    def _frame_gen_loop(self):
        spinner = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']
        spin_idx = 0
        start_time = time.time()
        while self.running:
            try:
                frame = next(self.frame_iter)
                # 若帧尺寸不对，resize到指定大小
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                # BGR转RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self.lock:
                    self.latest_frame = frame.copy()
                self.gen_frame_count += 1
                # 计算已生成时长
                elapsed = int(time.time() - start_time)
                h = elapsed // 3600
                m = (elapsed % 3600) // 60
                s = elapsed % 60
                time_str = f"{h:02d}:{m:02d}:{s:02d}"
                # 打印旋转光标+时长，动态刷新一行
                print(f"\r{spinner[spin_idx % len(spinner)]} 已生成: {time_str}", end='', flush=True)
                spin_idx += 1
            except StopIteration:
                self.running = False
                break
            except Exception as e:
                print(f"\n帧生成异常: {e}")
            time.sleep(1 / self.fps)

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.appsrc = rtsp_media.get_element().get_child_by_name('source')
        self.appsrc.connect('need-data', self.on_need_data)

    def on_need_data(self, src, length):
        with self.lock:
            frame = self.latest_frame.copy()
        # BGR转RGB（理论上_frame_gen_loop已转，这里保险再转一次）
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        data = frame.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.pts = buf.dts = int(self.number_frames * self.duration)
        buf.duration = int(self.duration)
        self.number_frames += 1
        retval = src.emit('push-buffer', buf)
        if retval != Gst.FlowReturn.OK:
            print('push buffer error:', retval)

class GstServer:
    def __init__(self, frame_generator):
        self.server = GstRtspServer.RTSPServer()
        factory = SensorFactory(frame_generator, fps=30)
        factory.set_shared(True)
        self.server.get_mount_points().add_factory("/video", factory)
        self.server.attach(None)

    def run(self):
        print("RTSP服务器已启动: rtsp://localhost:8554/video")
        loop = GLib.MainLoop()
        loop.run()

def main():
    print("启动RTSP视频流服务器...")
    # ===== 新增：初始化FrameGenerator =====
    inference_cfg = InferenceConfig()
    crop_cfg = CropConfig()
    pipeline = LivePortraitPipeline(inference_cfg, crop_cfg)
    args = ArgumentConfig(
        source="assets/examples/source/test.png",      # 修改为你的源图片路径
        driving="assets/examples/driving/d3.mp4",    # 修改为你的驱动视频路径
        output_dir="animations/"
    )
    frame_gen = FrameGenerator(pipeline, args)
    # 创建RTSP服务器
    server = GstServer(frame_gen)
    # 在后台线程中运行RTSP服务器
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    # 等待服务器启动
    time.sleep(3)
    print("开始推送人脸动画视频流...")
    print("按 Ctrl+C 停止服务器")
    print("使用以下命令测试RTSP流:")
    print("ffplay rtsp://localhost:8554/video")
    print("或")
    print("vlc rtsp://localhost:8554/video")
    try:
        while True:
            time.sleep(1/30)
    except KeyboardInterrupt:
        print("\n正在停止服务器...")
    finally:
        print("服务器已停止")

if __name__ == "__main__":
    main()
