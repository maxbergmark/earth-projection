#pragma once

#define PI 3.1415926536f
#define K_e 1.85407467730f

// coord.x = phi, coord.y = theta
float2 pixel_to_coord(float2 pos) {
	return (float2)(2.f*PI*pos.x, PI*pos.y);
}

float2 coord_to_pixel(float2 coord) {
	return (float2) (fmod(coord.x / (2.f*PI), 1.f), coord.y / PI);
}

float3 cart2sph(float3 pos) {
	float x = pos.x;
	float y = pos.y;
	float z = pos.z;
	float r = sqrt(x*x + y*y + z*z);
	float theta = atan2(sqrt(x*x+y*y), z);
	float phi = atan2(y, x);
	return (float3) (r, theta, phi);
}

float3 sph2cart(float3 sph) {
	float r = sph.x;
	float theta = sph.y;
	float phi = sph.z;
	return (float3)(
		r * cos(phi) * sin(theta),
		r * sin(phi) * sin(theta),
		r * cos(theta)
	);
}

float3 pixel_to_point_equirectangular(float2 pos) {
	float2 angles = pixel_to_coord(pos);
	float3 sph = (float3)(1, angles.y, angles.x);
	return sph2cart(sph);
}

float3 pixel_to_point_stereographic(float2 pos) {
	pos -= 0.5f;
	pos.x *= 4.f;
	pos.y *= 2.f;
	float x = pos.x;
	float y = pos.y;
	float xy = x*x + y*y;
	return (float3) (
		2.f*y / (1.f + xy),
		-2.f*x / (1.f + xy),
		(-1.f + xy) / (1.f + xy)
	);
}

float2 stretch_to_square(float2 pos) {
	float u = pos.x;
	float v = pos.y;
	if (u*u > v*v) {
		return (float2) (
			sign(u) * u*u / sqrt(u*u+v*v),
			sign(u) * u*v / sqrt(u*u+v*v)
		);
	} else {
		return (float2) (
			sign(v) * u*v / sqrt(u*u+v*v),
			sign(v) * v*v / sqrt(u*u+v*v)
		);
	}
}

float2 stretch_squircle(float2 pos) {
	float u = pos.x;
	float v = pos.y;
	float u2 = u*u;
	float v2 = v*v;
	return (float2) (
		u * sqrt(u2+v2-u2*v2) / sqrt(u2+v2),
		v * sqrt(u2+v2-u2*v2) / sqrt(u2+v2)
	);
}

float2 stretch_schwarz(float2 pos) {
	cfloat z = (cfloat) pos;
	cfloat temp = (cfloat) (.5f, .5f);
	cfloat u = cmult(K_e * temp, z);
	u.x -= K_e;
	cfloat cn = complex_cn(u, .5f);
	cfloat factor = ((cfloat) (1.f, -1.f)) / sqrt(2.f);
	cn = cmult(factor, cn);
	return (float2) cn;
}

float3 pixel_to_point_guyou(float2 pos) {
	pos -= 0.5f;
	pos.x *= 4.f;
	pos.y *= 2.f;

	if (pos.x > 0.f) {
		pos.x -= 1.0f;
		pos = stretch_schwarz(pos);
		float x = pos.x;
		float y = pos.y;
		float xy = x*x + y*y;
		return (float3) (
			2.f*x / (1.f + xy),
			(-1.f + xy) / (1.f + xy),
			-2.f*y / (1.f + xy)
		);
	} else {
		pos.x += 1.0f;
		pos = stretch_schwarz(pos);
		float x = pos.x;
		float y = pos.y;
		float xy = x*x + y*y;
		return (float3) (
			-2.f*x / (1.f + xy),
			-(-1.f + xy) / (1.f + xy),
			-2.f*y / (1.f + xy)
		);
	}
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