// Raymarched rendering of the odor field. The plume function is the one in Odor.cs (what the
// antennae sample); a 3D noise texture advected by the wind adds the billowing structure real
// plumes have (the antennae sample the smooth mean field — the noise is only visual). Drawn on a
// box covering the room, back faces only, marching from the camera (or the box entry) to the
// nearest opaque surface read from the depth buffer.
shader_type spatial;
render_mode unshaded, blend_mix, cull_front, depth_draw_never, depth_test_disabled, shadows_disabled, fog_disabled;

uniform sampler2D depth_tex : hint_depth_texture, filter_nearest;
uniform sampler3D noise_tex : filter_linear, repeat_enable;
uniform vec3 box_min = vec3(-12.0, 0.0, -9.0);
uniform vec3 box_max = vec3(12.0, 5.0, 9.0);
uniform vec3 wind = vec3(1.0, 0.0, 0.0);
uniform vec3 patch0 = vec3(0.0);
uniform vec3 patch1 = vec3(0.0);
uniform vec3 patch2 = vec3(0.0);
uniform int patch_count = 0;
uniform float plume_length = 18.0;
uniform float plume_width = 0.5;
uniform float cone_scale = 2.5;
uniform float wall_margin = 0.8;
uniform float end_margin = 1.5;
uniform float meander_amplitude = 1.3;
uniform float meander_wavelength = 6.0;
uniform float meander_period = 30.0;
uniform float time_s = 0.0;
uniform float core_height = 0.45;
uniform float vertical_width = 0.45;
uniform float vertical_meander_amplitude = 0.55;
uniform float vertical_meander_wavelength = 7.5;
uniform float rise_per_metre = 0.28;
uniform float density_scale = 0.12;
uniform float sparkle = 1.0;
uniform float clear_distance = 0.0;   // metres from the camera kept clear (the fly is at the end of it)
uniform float noise_scale = 0.28;
uniform float drift = 0.12;
uniform int steps = 48;
uniform vec3 tint : source_color = vec3(1.0, 0.8, 0.38);      // warm gold
uniform vec3 core_tint : source_color = vec3(1.0, 0.95, 0.75);
uniform vec3 sun_dir = vec3(0.4, 0.7, 0.5);

varying vec3 world_pos;

void vertex() {
	world_pos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
}


float along_to_wall(vec3 c) {
	float tx = wind.x > 1e-4 ? (box_max.x - c.x) / wind.x : (wind.x < -1e-4 ? (box_min.x - c.x) / wind.x : 1e9);
	float tz = wind.z > 1e-4 ? (box_max.z - c.z) / wind.z : (wind.z < -1e-4 ? (box_min.z - c.z) / wind.z : 1e9);
	return min(tx, tz);
}

float wall_factor(vec3 p) {
	float d = min(min(p.x - box_min.x, box_max.x - p.x), min(p.z - box_min.z, box_max.z - p.z)) / wall_margin;
	d = clamp(d, 0.0, 1.0);
	return d * d * (3.0 - 2.0 * d);
}

// Returns (axial strength, radial profile 0..1) so the renderer can draw a tube with an edge.
vec2 plume2(vec3 p, vec3 c) {
	vec3 d = p - c;
	float height = d.y;
	d.y = 0.0;
	float along = dot(d, wind);
	vec3 perp = vec3(-wind.z, 0.0, wind.x);
	float across = dot(d, perp);
	float phase = c.x * 0.7 + c.z * 1.3;
	float a = max(0.0, along);
	float grow = min(1.0, a / 3.0);
	float centre = meander_amplitude * grow * (sin(TAU * (a / meander_wavelength - time_s / meander_period) + phase)
	                                           + 0.45 * sin(TAU * (a / (meander_wavelength * 0.37) + time_s / (meander_period * 0.61)) + 2.1 * phase));
	float off = across - centre;
	float along_term = along > 0.0 ? -along / plume_length : -(along * along) / (2.0 * 1.2 * 1.2);
	float width = plume_width * (1.0 + a / cone_scale);
	float core_y = min(core_height + rise_per_metre * a + vertical_meander_amplitude * grow * sin(TAU * (a / vertical_meander_wavelength + time_s / (meander_period * 0.8)) + 3.7 * phase), box_max.y + 0.15 - 0.35 - c.y);
	float dh = height - core_y;
	float vw = vertical_width * (1.0 + a / cone_scale);
	float vertical = -(dh * dh) / (2.0 * vw * vw) - max(0.0, -height) * 4.0;
	float wall_cut = -max(0.0, a - along_to_wall(c) + end_margin) * 2.5;
	return vec2(exp(along_term + wall_cut) / sqrt(1.0 + a / cone_scale), exp(-(off * off) / (2.0 * width * width) + vertical));
}

// Tube density: a soft-edged tube around the centreline (the sensed field is the smooth Gaussian).
vec2 tube(vec3 p) {
	float dens = 0.0;
	float core = 0.0;
	if (patch_count > 0) { vec2 q = plume2(p, patch0); dens += q.x * smoothstep(0.12, 0.6, q.y); core = max(core, q.y); }
	if (patch_count > 1) { vec2 q = plume2(p, patch1); dens += q.x * smoothstep(0.12, 0.6, q.y); core = max(core, q.y); }
	if (patch_count > 2) { vec2 q = plume2(p, patch2); dens += q.x * smoothstep(0.12, 0.6, q.y); core = max(core, q.y); }
	return vec2(dens * wall_factor(p), core);
}

float hash13(vec3 q) {
	q = fract(q * vec3(0.1031, 0.1030, 0.0973));
	q += dot(q, q.yzx + 33.33);
	return fract((q.x + q.y) * q.z);
}

// Motes: a 3-D grid of 3.5 cm cells, each holding one mote at a hashed offset with a hashed size
// and flicker phase; ~one cell in twelve has a mote. Aperiodic: nothing repeats.
float glitter(vec3 p, vec3 rd) {
	vec3 cellp = p * 28.0;
	vec3 cell = floor(cellp);
	float h = hash13(cell);
	if (h < 0.92) { return 0.0; }
	vec3 centre = cell + vec3(hash13(cell + 1.7), hash13(cell + 5.3), hash13(cell + 9.1));
	float dist = length(cellp - centre);
	float mote = smoothstep(0.42, 0.05, dist);
	float phase = hash13(cell + 13.7) * TAU;
	float flicker = 0.35 + 0.65 * pow(0.5 + 0.5 * sin(TIME * (5.0 + 6.0 * hash13(cell + 2.9)) + phase + dot(rd, vec3(3.1, 5.7, 2.3)) * 4.0), 3.0);
	return mote * flicker;
}

float plume(vec3 p, vec3 c) {
	vec3 d = p - c;
	float height = d.y;
	d.y = 0.0;
	float along = dot(d, wind);
	vec3 perp = vec3(-wind.z, 0.0, wind.x);
	float across = dot(d, perp);
	float phase = c.x * 0.7 + c.z * 1.3;
	float a = max(0.0, along);
	float grow = min(1.0, a / 3.0);
	float centre = meander_amplitude * grow * (sin(TAU * (a / meander_wavelength - time_s / meander_period) + phase)
	                                           + 0.45 * sin(TAU * (a / (meander_wavelength * 0.37) + time_s / (meander_period * 0.61)) + 2.1 * phase));
	float off = across - centre;
	float along_term = along > 0.0 ? -along / plume_length : -(along * along) / (2.0 * 1.2 * 1.2);
	float width = plume_width * (1.0 + a / cone_scale);
	float dh = height - core_height;
	float vertical = -(dh * dh) / (2.0 * vertical_width * vertical_width) - max(0.0, -height) * 4.0;
	return exp(along_term - (off * off) / (2.0 * width * width) + vertical);
}

float field(vec3 p) {
	float c = 0.0;
	if (patch_count > 0) { c += plume(p, patch0); }
	if (patch_count > 1) { c += plume(p, patch1); }
	if (patch_count > 2) { c += plume(p, patch2); }
	return c;
}

vec2 box_hit(vec3 ro, vec3 rd) {
	vec3 inv = 1.0 / rd;
	vec3 t0 = (box_min - ro) * inv;
	vec3 t1 = (box_max - ro) * inv;
	vec3 tmin = min(t0, t1);
	vec3 tmax = max(t0, t1);
	return vec2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}

void fragment() {
	vec3 cam = INV_VIEW_MATRIX[3].xyz;
	vec3 rd = normalize(world_pos - cam);
	vec2 h = box_hit(cam, rd);
	float t0 = max(max(h.x, 0.0), clear_distance);
	float t1 = h.y;

	// Stop at the nearest opaque surface.
	float depth = texture(depth_tex, SCREEN_UV).r;
	vec3 ndc = vec3(SCREEN_UV * 2.0 - 1.0, depth);
	vec4 vp = INV_PROJECTION_MATRIX * vec4(ndc, 1.0);
	vp.xyz /= vp.w;
	vec3 scene = (INV_VIEW_MATRIX * vec4(vp.xyz, 1.0)).xyz;
	t1 = min(t1, length(scene - cam));
	if (t1 <= t0) {
		discard;
	}

	float n = float(steps);
	float dt = (t1 - t0) / n;
	float jitter = fract(sin(dot(SCREEN_UV * vec2(1234.5, 987.6), vec2(12.9898, 78.233)) + TIME * 0.7) * 43758.5453);
	float t = t0 + dt * jitter;
	float transmittance = 1.0;
	vec3 color = vec3(0.0);
	vec3 advect = wind * TIME * drift + vec3(0.0, TIME * 0.03, 0.0);
	for (int i = 0; i < 96; i++) {
		if (i >= steps) { break; }
		vec3 p = cam + rd * t;
		vec2 tb = tube(p);
		float c = max(0.0, tb.x - 0.03);   // the faint tail is smelled, not seen
		if (c > 0.0) {
			float nz = texture(noise_tex, p * noise_scale - advect).r;                 // 0..1, billows
			float nz2 = texture(noise_tex, p * noise_scale * 3.1 - advect * 1.7).r;   // finer wisps
			float structure = clamp(0.15 + 1.5 * nz * (0.5 + nz2), 0.0, 1.8);   // wisps and gaps
			c *= structure;
			float d = density_scale * c / (1.0 + c) * dt;
			float a = 1.0 - exp(-d);
			// Bright, slightly self-luminous core (> 1.0 feeds the glow), cooler translucent edge.
			// Glitter: motes at hashed positions (no tiling texture, so no grid), each with its own
			// flicker phase — bright enough to bloom, like dust catching the sun.
			float specks = glitter(p - advect * 2.3, rd) * sparkle;
			vec3 col = mix(tint, core_tint, tb.y * tb.y) * (0.9 + 1.4 * tb.y * nz) + vec3(1.0, 0.97, 0.85) * specks * 9.0;
			color += transmittance * a * col;
			transmittance *= 1.0 - a;
			if (transmittance < 0.02) { break; }
		}
		t += dt;
	}

	float alpha = 1.0 - transmittance;
	ALBEDO = color / max(alpha, 1e-4);
	ALPHA = alpha;
}
