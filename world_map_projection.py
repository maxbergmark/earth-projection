import numpy as np
import cv2
import math
# from scipy.spatial.transform.Rotation import from_rotvec
from scipy.spatial.transform import Rotation as R
import pyopencl as cl
import pyopencl.array as pycl_array
import time


class Renderer:

	def __init__(self):
		self.ctx = cl.create_some_context()
		self.queue = cl.CommandQueue(self.ctx, 
			properties=cl.command_queue_properties.PROFILING_ENABLE)

		self.program = cl.Program(self.ctx, open("projection.cl", "r").read()).build()  # Create the OpenCL program
		self.read_images()
		# img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
		self.h, self.w = (500, 1000)

		fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
		self.dest_buf = cl.Image(self.ctx, cl.mem_flags.WRITE_ONLY, fmt, shape=(self.w, self.h))
		self.dest = np.empty((self.h, self.w, 4), dtype=np.uint8)
		self.angle_x = 0
		self.angle_y = 0
		self.angle_z = 0
		cv2.namedWindow("image")

	def read_images(self):
		earth_img = cv2.imread('8081_earthmap10k.jpg', cv2.IMREAD_UNCHANGED)
		earth_img = cv2.cvtColor(earth_img, cv2.COLOR_BGR2BGRA)
		night_img = cv2.imread('8081_earthlights10k.jpg', cv2.IMREAD_UNCHANGED)
		night_img = cv2.cvtColor(night_img, cv2.COLOR_BGR2BGRA)
		self.earth_src_buf = cl.image_from_array(self.ctx, earth_img, 4)
		self.night_src_buf = cl.image_from_array(self.ctx, night_img, 4)



	def update_angles(self, event, x, y, flags, param):
		self.angle_x = x / self.w * 2 * np.pi
		self.angle_y = (y / self.h - 0.5) * 2 * np.pi

	def render_image(self):
		self.program.project(self.queue, (self.w, self.h), None, 
			self.earth_src_buf, self.night_src_buf, self.dest_buf, 
			np.float32(self.angle_z), np.float32(self.angle_y), np.float32(-self.angle_x))
		cl.enqueue_copy(self.queue, self.dest, self.dest_buf, origin=(0, 0), region=(self.w, self.h))

	def save_animation(self):
		fourcc = cv2.VideoWriter_fourcc(*'avc1') 
		video=cv2.VideoWriter('video.mp4', fourcc, 60.0, (self.w, self.h))
		n_frames = 400
		for i in range(n_frames):
			self.angle_x = i / n_frames * 8 * np.pi
			self.angle_y = i / n_frames * 4 * np.pi
			self.angle_z = i / n_frames * 2 * np.pi
			self.render_image()
			cv2.imshow('image', self.dest)
			video.write(self.dest[:,:,:3])
			if cv2.waitKey(10) == ord('q'):
				break

		cv2.destroyAllWindows()
		video.release()


	def show_animation(self):
		cv2.setMouseCallback('image', self.update_angles)
		last = time.perf_counter()
		while True:
			t0 = time.perf_counter()
			self.render_image()
			t1 = time.perf_counter()
			# print(f"{t1-t0:.3f}, {t1-last:.3f}")
			last = t1
			# img = cv2.resize(self.dest, (1000, 500), interpolation = cv2.INTER_AREA)
			img = self.dest
			cv2.imshow('image', img)
			if cv2.waitKey(10) == ord('q'):
				break

renderer = Renderer()
renderer.show_animation()
# renderer.save_animation()