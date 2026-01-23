@group(0) @binding(0) var out: texture_storage_2d<rgba32float, write>;

struct primitive {
    ptype: u32,
    inv_transform: mat4x4f,
}
struct input {
    camera: mat4x4f,
    fov: f32,
    primitive_count: u32, 
}

@group(0) @binding(1) var<uniform> uniform: input;
@group(0) @binding(2) var<storage, read> primitives: primitive;

struct hit {
    pos: vec3f,
    normal: vec3f,
    t: f32,
}
struct ray {
    orig: vec3f,
    dir: vec3f,
}
fn transform_ray(r: ptr<function, ray>, mat: mat4x4f) {
    let o = mat * vec4f(r.orig, 1);
    r.orig = o.xyz / o.w;
    let d = mat * vec4f(r.dir, 0);
    r.dir = d.xyz;
}
fn at(r: ray, t: f32) -> vec3f {
    return r.orig + t * r.dir;
}

fn sky_color(dir: vec3f) -> vec3f {
    return mix(vec3f(1), vec3f(0.5, 0.7, 1.0), dir.y / 2 + 0.5);
}

fn intersect_box(a: ray, transform: mat4x4f) -> hit {
    var r = a;
    transform_ray(&r, transform);

    let box_min = vec3f(-1.0);
    let box_max = vec3f(1.0);

    let inv_dir = 1.0 / r.dir;

    let t0 = (box_min - r.orig) * inv_dir;
    let t1 = (box_max - r.orig) * inv_dir;

    let tmin = max(
        max(min(t0.x, t1.x), min(t0.y, t1.y)),
        min(t0.z, t1.z)
    );

    let tmax = min(
        min(max(t0.x, t1.x), max(t0.y, t1.y)),
        max(t0.z, t1.z)
    );

    if tmax < 0.0 || tmin > tmax {
        return hit(vec3f(0), vec3f(0), 0);
    }
    
    let pos = at(r, tmin);
    let pos_m = abs(pos);
    let pos_max = max(max(pos_m.x, pos_m.y), pos_m.z);
    if pos_max == pos_m.x {
        return hit(at(a, tmin), vec3f(pos.x, 0, 0), tmin);
    } else if pos_max == pos_m.y {
        return hit(at(a, tmin), vec3f(0, pos.y, 0), tmin);
    } else {
        return hit(at(a, tmin), vec3f(0, 0, pos.z), tmin);
    }
}
fn intersect_sphere(x: ray, transform: mat4x4f) -> hit {
    var r = x;
    transform_ray(&r, transform);

    let oc = -r.orig;
    let a = dot(r.dir, r.dir);
    let b = -2 * dot(r.dir, oc);
    let c = dot(oc, oc) - 1;

    let discriminant = b*b - 4*a*c;
    if (discriminant >= 0) {
        let dist = (-b - sqrt(discriminant)) / (2*a);
        if dist > 0 {
            let normal = normalize(at(r, dist));
            return hit(at(x, dist), normal, dist);
        }
    }
    return hit(vec3f(0), vec3f(0), 0);
}

fn ray_color(r: ray) -> vec3f {
    let hit_a = intersect_sphere(
        r,
        mat4x4f(
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 0.5, 0,
            0, 0, 0, 1,
        )
    );
    let k = sqrt(2.0)/2.0;
    let transform = mat4x4f(
        k, 0, -k, 0,
        0, 1, 0, 0,
        k, 0, k, 0,
        0, 0, 0, 1,
    );
    var hit = intersect_box(r, transform);

    if hit.t <= 0 || (hit.t > hit_a.t && hit_a.t > 0) {
        hit = hit_a;
    }
    if hit.t != 0 {
        return vec3f(dot(hit.normal, -normalize(vec3f(-1, -3, 0)))) / 4 + vec3f(0.75);
    }
    return sky_color(normalize(r.dir));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let pos = id.xy;
    let size = textureDimensions(out);
    if (pos.x >= size.x || pos.y >= size.y) {
        return;
    }
    let ar = f32(size.x) / f32(size.y);
    var uv = (vec2f(pos) / vec2f(size) - vec2f(0.5)) * 2;
    uv.y = -uv.y;
    let dir = vec3f(tan(uniform.fov/2) * uv.x, tan(uniform.fov/2) / ar * uv.y, 1);
    var r = ray(vec3f(0), dir);
    transform_ray(&r, uniform.camera);
    textureStore(out, pos, vec4f(ray_color(r), 1.0));
}
