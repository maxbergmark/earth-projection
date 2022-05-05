#include "jacobi.c"
#include "complex.c"
#include "mappings.c"

__constant float4 white = (float4)(1.f, 1.f, 1.f, 1.f);
__constant float4 black = (float4)(0.f, 0.f, 0.f, 1.f);

float get_angle(float3 v1, float3 v2) {
	return acos(dot(v1, v2) / (length(v1) * length(v2)));
}

float get_grid_opacity(float2 pos) {
	// float grid_size = PI/60.f;
	pos.x += 0.5f;
	float grid_y = 1.f/18.f;
	float grid_x = .5f*grid_y;
	float o = min(fabs(fmod(pos.x + .5f*grid_x, grid_x) - .5f*grid_x)
		, fabs(fmod(pos.y + .5f*grid_y, grid_y) - .5f*grid_y));
	o /= grid_x;
	o = 1.f - o;

	return pow(clamp(o, 0.0f, 1.0f), 100.f);
}

float4 add_grid(float4 pix, float4 grid_color, float2 pos) {
	float grid_opacity = get_grid_opacity(pos);
	return mix(pix, grid_color, grid_opacity);
}

float4 mix_night(float4 earth_pix, float4 night_pix, float3 point, float3 sun_pos) {
	float sun_angle = get_angle(point, sun_pos);
	float clamped = clamp((sun_angle - .5f*PI)*5.f + .5f, 0.f, 1.f);

	return mix(earth_pix, night_pix, clamped);
}

__kernel void project(read_only image2d_t earth_src, read_only image2d_t night_src, 
	write_only image2d_t dest, float rot_x, float rot_y, float rot_z, float t) {


	const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR;
	int2 int_pos = (int2)(get_global_id(0), get_global_id(1));
	int2 size = (int2)(get_global_size(0), get_global_size(1));
	
	float2 pos = (float2)((float) int_pos.x / size.x, (float) int_pos.y / size.y);


	// float3 point = pixel_to_point_equirectangular(pos);
	float3 point = pixel_to_point_guyou(pos);
	// float3 point = pixel_to_point_stereographic(pos);
	float3 rotated_point = rotate_point(point, rot_x, rot_y, rot_z);
	float3 rotated_sph = cart2sph(rotated_point);
	float2 rotated_pos = coord_to_pixel(rotated_sph.zy);


	float4 earth_pix = read_imagef(earth_src, sampler, rotated_pos);
	float4 night_pix = read_imagef(night_src, sampler, rotated_pos);
	earth_pix = add_grid(earth_pix, black, rotated_pos);
	night_pix = add_grid(night_pix, white, rotated_pos);

	float3 sun_pos = (float3) (cos(t), sin(t), 0.3);
	earth_pix = mix_night(earth_pix, night_pix, rotated_point, sun_pos);

	write_imagef(dest, int_pos, earth_pix);
    // write_imagef(dest, int_pos, mix_pix);
}

__kernel void downscale(read_only image2d_t src, 
	write_only image2d_t dest, int scale) {

	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;
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