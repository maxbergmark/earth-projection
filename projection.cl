
#define PI 3.1415926536

// coord.x = phi, coord.y = theta
float2 pixel_to_coord(float2 pos) {
	return (float2)(2*PI*pos.x, PI*pos.y);
}

float2 coord_to_pixel(float2 coord) {
	return (float2) (fmod(coord.x / (2*PI), 1.0), coord.y / PI);
}

float3 pixel_to_pos_stereographic(float2 pos) {
	float x = pos.x;
	float y = pos.y;
	float xy = x*x + y*y;
	return (float3) (
		2*x / (1 + xy),
		2*y / (1 + xy),
		(-1 + xy) / (1 + xy)
	);
}

float3 pixel_to_pos_guyou(float2 pos) {
	if (pos.x > 0) {
		pos.x -= 1.0;
		float a = atan2(pos.x, pos.y);
		pos *= 1 - 0.2f * fabs(sin(2.f*a));
		float x = pos.x;
		float y = pos.y;
		float xy = x*x + y*y;
		return (float3) (
			2*x / (1 + xy),
			(-1 + xy) / (1 + xy),
			-2*y / (1 + xy)
		);
	} else {
		pos.x += 1.0;
		float a = atan2(pos.x, pos.y);
		pos *= 1 - 0.2f * fabs(sin(2.f*a));
		float x = pos.x;
		float y = pos.y;
		float xy = x*x + y*y;
		return (float3) (
			-2*x / (1 + xy),
			-(-1 + xy) / (1 + xy),
			-2*y / (1 + xy)
		);
	}
}


float get_angle(float3 v1, float3 v2) {
	return acos(dot(v1, v2) / (length(v1) * length(v2)));
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

__kernel void project(read_only image2d_t earth_src, read_only image2d_t night_src, 
	write_only image2d_t dest, float rot_x, float rot_y, float rot_z) {

	const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR;
	int2 int_pos = (int2)(get_global_id(0), get_global_id(1));
	int2 size = (int2)(get_global_size(0), get_global_size(1));
	
	float2 pos = (float2)((float) int_pos.x / size.x, (float) int_pos.y / size.y);
	float2 angles = pixel_to_coord(pos);
	float3 sph = (float3)(1, angles.y, angles.x);
	float3 point = sph2cart(sph);

	pos.x -= 0.5;
	pos.y -= 0.5;

	pos.x *= 4;
	pos.y *= 2;
	// if (min(distance(pos, (float2)(-1, 0.0)), distance(pos, (float2)(1, 0.0))) > 1.0) {
		// return;
	// }
	point = pixel_to_pos_guyou(pos);
	if (point.y < 0 && pos.x < 0) {
	    write_imagef(dest, int_pos, (float4) (0, 0, 0, 0));
		return;		
	}

	if (point.y > 0 && pos.x > 0) {
	    write_imagef(dest, int_pos, (float4) (0, 0, 0, 0));
		return;		
	}

	float3 rotated_point = rotate_around_x(point, rot_x);
	rotated_point = rotate_around_y(rotated_point, rot_y);
	// rotated_point = point;
	rotated_point = rotate_around_z(rotated_point, rot_z);
	float3 rotated_sph = cart2sph(rotated_point);
	float2 rotated_pos = coord_to_pixel(rotated_sph.zy);
	if (fmod(fabs(rotated_point.z), 0.1f) < 0.005f) {
	    write_imagef(dest, int_pos, (float4) (0, 0, 0, 0));
		return;
	}

	// if (fmod(fabs(rotated_point.x), 0.1f) < 0.005f) {
	    // write_imagef(dest, int_pos, (float4) (0, 0, 0, 0));
		// return;
	// }

	float4 earth_pix = read_imagef(earth_src, sampler, rotated_pos);
	float4 night_pix = read_imagef(night_src, sampler, rotated_pos);
	float3 sun_pos = (float3) (cos(0.5*rot_z), sin(0.5*rot_z), 0.3);
	float sun_angle = get_angle(rotated_point, sun_pos);
	float clamped = clamp((sun_angle - PI / 2)*5 + 0.5, 0.0, 1.0);

	float4 mix_pix = mix(earth_pix, night_pix, clamped);
    write_imagef(dest, int_pos, earth_pix);
    // write_imagef(dest, int_pos, mix_pix);
}
