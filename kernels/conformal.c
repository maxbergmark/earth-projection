#pragma once

#include "kernels/jacobi.c"
#include "kernels/complex.c"

#define PI 3.1415926536f
#define K_e 1.85407467730f

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

float2 rotate_pos(float2 pos, float angle) {
	return (float2) (
		cos(angle)*pos.x - sin(angle)*pos.y, 
		sin(angle)*pos.x + cos(angle)*pos.y
	);
}

int get_quadrant(float2 pos) {
	float a = atan2(pos.x, pos.y);
	a *= 2.f/PI;
	a += 2.f;
	return ((int) a + 5) % 4;
}

float3 pixel_to_point_peirce(float2 pos) {
	pos -= 0.5f;
	pos.x *= 2.f*sqrt(2.f);
	pos.y *= 2.f*sqrt(2.f);

	if (fabs(pos.x) + fabs(pos.y) < 1.f*sqrt(2.f)) {
		// northern hemisphere
		pos = rotate_pos(pos, 0.25f*PI);
		pos = stretch_schwarz(pos);
		pos = rotate_pos(pos, -0.25f*PI);
		float x = pos.x;
		float y = pos.y;
		float xy = x*x + y*y;
		return (float3) (
			-2.f*y / (1.f + xy),
			-2.f*x / (1.f + xy),
			-(-1.f + xy) / (1.f + xy)
		);
	} else {
		// southern hemisphere
		int q = get_quadrant(pos);
		float xf = q % 3 > 0 ? 1.0f : -1.0f;
		float yf = q < 2 ? 1.0f : -1.0f;
		float2 offset = (float2) (xf * sqrt(2.f), yf * sqrt(2.f));
		pos += offset;
		pos = rotate_pos(pos, 0.25f*PI);
		if (q % 2 == 1) {
			pos = rotate_pos(pos, PI);
		}
		pos = stretch_schwarz(pos);
		pos = rotate_pos(pos, -0.25f*PI);

		float x = pos.x;
		float y = pos.y;
		float xy = x*x + y*y;
		return (float3) (
			-2.f*x / (1.f + xy),
			-2.f*y / (1.f + xy),
			(-1.f + xy) / (1.f + xy)
		);
	}
}