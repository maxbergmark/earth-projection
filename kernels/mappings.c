#pragma once

#include "kernels/conformal.c"

#define PI 3.1415926536f

float get_angle(float3 v1, float3 v2) {
	return acos(dot(v1, v2) / (length(v1) * length(v2)));
}

float get_sum(float4 v) {
	return v.x + v.y + v.z + v.w;
}

// coord.x = phi, coord.y = theta
float2 pixel_to_coord(float2 pos) {
	return (float2)(2.f*PI*pos.x, PI*pos.y);
}

float2 coord_to_pixel(float2 coord) {
	return (float2) (fmod(coord.x / (2.f*PI), 1.f), coord.y / PI);
}

float3 cartesian_to_spherical(float3 pos) {
	float x = pos.x;
	float y = pos.y;
	float z = pos.z;
	float r = sqrt(x*x + y*y + z*z);
	float theta = atan2(sqrt(x*x+y*y), z);
	float phi = atan2(y, x);
	return (float3) (r, theta, phi);
}

float3 spherical_to_cartesian(float3 sph) {
	float r = sph.x;
	float theta = sph.y;
	float phi = sph.z;
	return (float3)(
		r * cos(phi) * sin(theta),
		r * sin(phi) * sin(theta),
		r * cos(theta)
	);
}

float3 tangent_to_world_space(float3 p, float3 normal) {
	float3 p_t = normalize(cross((float3) (0, 0, 1), p));
	float3 p_z = normalize(cross(p, p_t));
	return normal.x * p + normal.y * p_t + normal.z * p_z;
}

float3 pixel_to_point_equirectangular(float2 pos) {
	float2 angles = pixel_to_coord(pos);
	float3 sph = (float3)(1, angles.y, angles.x);
	return spherical_to_cartesian(sph);
}

float2 point_to_pixel_equirectangular(float3 point) {
	float3 sph = cartesian_to_spherical(point);
	return coord_to_pixel(sph.zy);
}

float3 pixel_to_point_stereographic(float2 pos, int2 size) {
	pos -= 0.5f;
	pos *= 0.001f * (float2) (size.x, size.y);
	float x = pos.x;
	float y = pos.y;
	float xy = x*x + y*y;
	return (float3) (
		-2.f*y / (1.f + xy),
		-2.f*x / (1.f + xy),
		-(-1.f + xy) / (1.f + xy)
	);
}

float3 rotate_around_x(float3 pos, float angle) {
	return (float3)(
		pos.x, 
		cos(angle) * pos.y - sin(angle) * pos.z,
		sin(angle) * pos.y + cos(angle) * pos.z
	);
}

float3 rotate_around_y(float3 pos, float angle) {
	return (float3)(
		cos(angle) * pos.x + sin(angle) * pos.z,
		pos.y, 
		-sin(angle) * pos.x + cos(angle) * pos.z
	);
}

float3 rotate_around_z(float3 pos, float angle) {
	return (float3)(
		cos(angle) * pos.x - sin(angle) * pos.y,
		sin(angle) * pos.x + cos(angle) * pos.y,
		pos.z
	);
}

float3 rotate_point(float3 point, float rot_x, float rot_y, float rot_z) {
	float3 rotated_point = rotate_around_x(point, rot_x);
	rotated_point = rotate_around_y(rotated_point, rot_y);
	rotated_point = rotate_around_z(rotated_point, rot_z);
	return rotated_point;
}