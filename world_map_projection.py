import numpy as np
import cv2
import pyopencl as cl
import time

class Renderer:

	def __init__(self):
		self.window_name = "Map Projection"
		cv2.namedWindow(self.window_name)

		self.upscale = 2
		self.h, self.w = (1200, 2400)
		self.render_h = self.h*self.upscale
		self.render_w = self.w*self.upscale
		self.angle_x, self.angle_y, self.angle_z, self.t = 0, 0, 0, 0
		self.setup_opencl()


	def setup_opencl(self):
		self.ctx = cl.create_some_context()
		self.queue = cl.CommandQueue(self.ctx, 
			properties=cl.command_queue_properties.PROFILING_ENABLE)

		self.program = cl.Program(self.ctx, 
			open("kernels/projection.cl", "r").read()).build()
		self.read_images()

		fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
		self.render_buf = cl.Image(self.ctx, 
			cl.mem_flags.READ_WRITE, fmt, shape=(self.render_w, self.render_h))
		self.render = np.empty(
			(self.render_h, self.render_w, 4), dtype=np.float32)

		self.view_buf = cl.Image(self.ctx, 
			cl.mem_flags.READ_WRITE, fmt, shape=(self.w, self.h))
		self.view = np.empty((self.h, self.w, 4), dtype=np.float32)

	def read_image(self, filename):
		print("Reading texture:", filename)
		img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
		img = img.astype(np.float32) / 255
		return cl.image_from_array(self.ctx, img, 4)

	def read_images(self):
		self.earth_buf = self.read_image('textures/8081_earthmap10k.jpg')
		self.specular_buf = self.read_image('textures/8081_earthspec10k.jpg')
		self.normal_buf = self.read_image('textures/8k_earth_normal_map.jpg')
		self.night_buf = self.read_image('textures/8081_earthlights10k.jpg')

	def update_angles(self, event, x, y, flags, param):
		self.angle_x = x / self.w * 2 * np.pi
		self.angle_y = (y / self.h - 0.5) * 2 * np.pi
		# self.angle_y = self.angle_x

	def render_image(self):
		self.program.project(self.queue, (self.render_w, self.render_h), None, 
			self.earth_buf, self.night_buf, 
			self.specular_buf, self.normal_buf, self.render_buf, 
			np.float32(self.angle_z), np.float32(self.angle_y), 
			np.float32(-self.angle_x), np.float32(self.t))

		self.program.downscale(self.queue, (self.w, self.h), None, 
			self.render_buf, self.view_buf, np.int32(self.upscale))
		
		# cl.enqueue_copy(self.queue, self.dest, self.dest_buf, 
			# origin=(0, 0), region=(self.render_w, self.render_h))
		cl.enqueue_copy(self.queue, self.view, self.view_buf, 
			origin=(0, 0), region=(self.w, self.h))

	def save_animation(self):
		fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
		video=cv2.VideoWriter('video.mp4', fourcc, 60.0, (self.w, self.h))

		fmt_rgba = cl.ImageFormat(cl.channel_order.RGBA, 
			cl.channel_type.UNSIGNED_INT8)
		frame_buf = cl.Image(self.ctx, 
			cl.mem_flags.READ_WRITE, fmt_rgba, shape=(self.w, self.h))
		frame_rgba = np.empty((self.h, self.w, 4), dtype=np.uint8)



		n_frames = 1600
		for i in range(n_frames):
			print(f"\r{i:5d} / {n_frames:5d}", end="", flush=True)
			self.angle_x = i / n_frames * 8 * np.pi
			self.angle_y = i / n_frames * 4 * np.pi
			self.angle_z = i / n_frames * 2 * np.pi
			self.t = i / n_frames * 2 * np.pi
			self.render_image()
			cv2.imshow(self.window_name, self.view)
			self.program.float_to_uint8(self.queue, (self.w, self.h), None, 
				self.view_buf, frame_buf)
			cl.enqueue_copy(self.queue, frame_rgba, frame_buf, 
				origin=(0, 0), region=(self.w, self.h))

			video.write(frame_rgba[:,:,:3])
			if cv2.waitKey(1) == ord('q'):
				break

		print()
		cv2.destroyAllWindows()
		video.release()

	def show_animation(self):
		cv2.setMouseCallback(self.window_name, self.update_angles)
		last = time.perf_counter()
		while True:
			t0 = time.perf_counter()
			self.t = .5*time.time() % (2*np.pi)
			self.render_image()
			t1 = time.perf_counter()
			print(f"\r{t1-t0:.3f}, {t1-last:.3f}", end="", flush=True)
			last = t1
			cv2.imshow(self.window_name, self.view)
			if cv2.waitKey(1) == ord('q'):
				break

		print()


renderer = Renderer()
renderer.show_animation()
# renderer.save_animation()