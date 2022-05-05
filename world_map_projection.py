import numpy as np
import cv2
import pyopencl as cl
import time

class Renderer:

	def __init__(self):
		self.ctx = cl.create_some_context()
		self.window_name = "Map Projection"
		self.queue = cl.CommandQueue(self.ctx, 
			properties=cl.command_queue_properties.PROFILING_ENABLE)

		self.program = cl.Program(self.ctx, open("projection.cl", "r").read()).build()  # Create the OpenCL program
		self.read_images()
		# img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
		self.h, self.w = (1000, 2000)
		self.upscale = 2
		self.render_h, self.render_w = (self.h*self.upscale, self.w*self.upscale)

		fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
		self.render_buf = cl.Image(self.ctx, cl.mem_flags.READ_WRITE, fmt, shape=(self.render_w, self.render_h))
		self.render = np.empty((self.render_h, self.render_w, 4), dtype=np.float32)

		self.view_buf = cl.Image(self.ctx, cl.mem_flags.WRITE_ONLY, fmt, shape=(self.w, self.h))
		self.view = np.empty((self.h, self.w, 4), dtype=np.float32)

		self.angle_x, self.angle_y, self.angle_z = 0, 0, 0
		cv2.namedWindow(self.window_name)

	def read_images(self):
		earth_img = cv2.imread('8081_earthmap10k.jpg', cv2.IMREAD_UNCHANGED)
		earth_img = cv2.cvtColor(earth_img, cv2.COLOR_BGR2BGRA)
		earth_img = earth_img.astype(np.float32)
		earth_img /= 255
		night_img = cv2.imread('8081_earthlights10k.jpg', cv2.IMREAD_UNCHANGED)
		night_img = cv2.cvtColor(night_img, cv2.COLOR_BGR2BGRA)
		night_img = night_img.astype(np.float32)
		night_img /= 255
		self.earth_src_buf = cl.image_from_array(self.ctx, earth_img, 4)
		self.night_src_buf = cl.image_from_array(self.ctx, night_img, 4)

	def update_angles(self, event, x, y, flags, param):
		self.angle_x = x / self.w * 2 * np.pi
		self.angle_y = (y / self.h - 0.5) * 2 * np.pi
		# self.angle_y = self.angle_x

	def render_image(self):
		self.program.project(self.queue, (self.render_w, self.render_h), None, 
			self.earth_src_buf, self.night_src_buf, self.render_buf, 
			np.float32(self.angle_z), np.float32(self.angle_y), 
			np.float32(-self.angle_x), np.float32(.5*time.time() % (2*np.pi)))

		self.program.downscale(self.queue, (self.w, self.h), None, 
			self.render_buf, self.view_buf, np.int32(self.upscale))
		
		# cl.enqueue_copy(self.queue, self.dest, self.dest_buf, origin=(0, 0), region=(self.render_w, self.render_h))
		cl.enqueue_copy(self.queue, self.view, self.view_buf, origin=(0, 0), region=(self.w, self.h))

	def save_animation(self):
		fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
		video=cv2.VideoWriter('video.mp4', fourcc, 60.0, (self.w, self.h))

		n_frames = 400
		for i in range(n_frames):
			self.angle_x = i / n_frames * 8 * np.pi
			self.angle_y = i / n_frames * 4 * np.pi
			self.angle_z = i / n_frames * 2 * np.pi
			self.render_image()
			cv2.imshow(self.window_name, self.dest)
			frame_buf = (self.view_buf[:,:,:3]*255).astype(np.uint8)
			video.write(frame_buf)
			if cv2.waitKey(1) == ord('q'):
				break

		cv2.destroyAllWindows()
		video.release()

	def show_animation(self):
		cv2.setMouseCallback(self.window_name, self.update_angles)
		last = time.perf_counter()
		while True:
			t0 = time.perf_counter()
			self.render_image()
			t1 = time.perf_counter()
			print(f"{t1-t0:.3f}, {t1-last:.3f}")
			last = t1
			cv2.imshow(self.window_name, self.view)
			if cv2.waitKey(1) == ord('q'):
				break


renderer = Renderer()
renderer.show_animation()
# renderer.save_animation()