#include "kernels/mappings.c"
#include "kernels/conversion.c"

__constant float4 white = (float4)(1.f, 1.f, 1.f, 1.f);
__constant float4 black = (float4)(0.f, 0.f, 0.f, 1.f);

float get_grid_opacity(float2 pos) {
	pos.x += 0.5f;
	float grid_y = 1.f/18.f;
	float grid_x = .5f*grid_y;
	float o = min(fabs(fmod(pos.x + .5f*grid_x, grid_x) - .5f*grid_x)
		, fabs(fmod(pos.y + .5f*grid_y, grid_y) - .5f*grid_y));
	o = 1.f - o / grid_x;

	return pow(clamp(o, 0.0f, 1.0f), 100.f);
}

float4 add_grid(float4 pix, float4 grid_color, float2 pos) {
	float grid_opacity = get_grid_opacity(pos);
	return mix(pix, grid_color, grid_opacity);
}

float4 mix_night(float4 earth_pix, float4 night_pix, 
	float3 point, float3 sun_pos, float3 normal) {
	
	float sun_angle = get_angle(point, sun_pos);
	float clamped = clamp((sun_angle - .5f*PI)*5.f + .5f, 0.f, 1.f);

	float3 world_normal = tangent_to_world_space(point, normal);
	// return (float4) (world_normal.x, world_normal.y, world_normal.z, 1.f);
	float factor = get_angle(world_normal, sun_pos) / (.5f*PI);
	earth_pix *= 1.f + .2f*(1.f-factor);

	return mix(earth_pix, night_pix, clamped);
}

__kernel void project(read_only image2d_t earth_src, 
	read_only image2d_t night_src, read_only image2d_t specular_src, 
	read_only image2d_t normal_src, write_only image2d_t dest, 
	float rot_x, float rot_y, float rot_z, float t, float4 proj_mix) {


	const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE 
		| CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR;
	int2 int_pos = (int2)(get_global_id(0), get_global_id(1));
	int2 size = (int2)(get_global_size(0), get_global_size(1));
	
	float2 pos = (float2)((float) int_pos.x / size.x, (float) int_pos.y / size.y);

	float3 point_equi = pixel_to_point_equirectangular(pos);
	float3 point_stereo = pixel_to_point_stereographic(pos, size);
	float3 point_guyou = pixel_to_point_guyou(pos);
	float3 point_peirce = pixel_to_point_peirce(pos);

	proj_mix /= get_sum(proj_mix);
	float3 point = normalize(
		proj_mix.x * point_equi
		+ proj_mix.y * point_stereo
		+ proj_mix.z * point_peirce
		+ proj_mix.w * point_guyou
	);

	// write_imagef(dest, int_pos, (float4) (point.x, point.y, point.z, 1.f));
	// return;

	float3 rotated_point = rotate_point(point, rot_x, rot_y, rot_z);
	// float3 rotated_point = point;
	float2 rotated_pos = point_to_pixel_equirectangular(rotated_point);


	float4 earth_pix = read_imagef(earth_src, sampler, rotated_pos);
	float4 night_pix = read_imagef(night_src, sampler, rotated_pos);
	float specularity = read_imagef(specular_src, sampler, rotated_pos).x;
	float4 normal_pix = read_imagef(normal_src, sampler, rotated_pos);
	float3 normal = 2.f*normal_pix.xyz - 1.f;

	earth_pix = add_grid(earth_pix, black, rotated_pos);
	night_pix = add_grid(night_pix, white, rotated_pos);

	float3 sun_pos = normalize((float3) (cos(t), sin(t), 0.3));
	earth_pix = mix_night(earth_pix, night_pix, rotated_point, sun_pos, normal);

	write_imagef(dest, int_pos, earth_pix);
}


