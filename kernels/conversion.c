#pragma once

__kernel void downscale(read_only image2d_t src, 
	write_only image2d_t dest, int scale) {

	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE 
		| CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
	int2 int_pos = (int2)(get_global_id(0), get_global_id(1));
	float factor = 1.f / (scale*scale);

	float4 res = (float4) (0.f);
	for (int i = 0; i < scale; i++) {
		for (int j = 0; j < scale; j++) {
			int2 offset = (int2) (i, j);
			res += read_imagef(src, sampler, scale*int_pos + offset);
		}
	}
	write_imagef(dest, int_pos, factor*res);
}

__kernel void float_to_uint8(read_only image2d_t src, 
	write_only image2d_t dest) {

	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE 
		| CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
	int2 int_pos = (int2)(get_global_id(0), get_global_id(1));

	float4 p = read_imagef(src, sampler, int_pos);
	uint4 uint_pix = (uint4) (p.x*255, p.y*255, p.z*255, p.w*255);
	write_imageui(dest, int_pos, uint_pix);
}

__kernel void uint8_to_float(read_only image2d_t src, 
	write_only image2d_t dest) {

	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE 
		| CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
	int2 int_pos = (int2)(get_global_id(0), get_global_id(1));

	uint4 p = read_imageui(src, sampler, int_pos);
	float4 float_pix = (float4) (p.x, p.y, p.z, p.w) / 255.f;
	write_imagef(dest, int_pos, float_pix);
}