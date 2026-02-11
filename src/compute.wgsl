@group(1) @binding(0) var out: texture_storage_2d<rgba32float, write>;

struct primitive {
    ptype: u32,
    transform: mat4x4f,
    inv_transform: mat4x4f,
}
struct portal_pair {
    transform_a: mat4x4f,
    inv_transform_a: mat4x4f,
    transform_b: mat4x4f,
    inv_transform_b: mat4x4f,
}
struct input {
    camera: mat4x4f,
    fov: f32,
    primitive_count: u32, 
    portal_pair_count: u32,
}

@group(0) @binding(0) var<uniform> uniform: input;
@group(0) @binding(1) var<storage, read> primitives: array<primitive>;
@group(0) @binding(2) var<storage, read> portals: array<portal_pair>;

struct hit {
    pos: vec3f,
    normal: vec3f,
    t: f32,
}
struct portal_hit {
    pos: vec3f,
    uv: vec2f,
    t: f32,
    to_other: mat4x4f,
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

fn intersect_box(a: ray, transform: mat4x4f, inv_transform: mat4x4f) -> hit {
    var r = a;
    transform_ray(&r, inv_transform);

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
    
    var normal: vec3f;
    if pos_max == pos_m.x {
        normal = vec3f(pos.x, 0, 0);
    } else if pos_max == pos_m.y {
        normal = vec3f(0, pos.y, 0);
    } else {
        normal = vec3f(0, 0, pos.z);
    }

    return hit(at(a, tmin), (transform * vec4f(normal, 0)).xyz, tmin);
}
fn intersect_sphere(x: ray, transform: mat4x4f, inv_transform: mat4x4f) -> hit {
    var r = x;
    transform_ray(&r, inv_transform);

    let oc = -r.orig;
    let a = dot(r.dir, r.dir);
    let b = -2 * dot(r.dir, oc);
    let c = dot(oc, oc) - 1;

    let discriminant = b*b - 4*a*c;
    if discriminant >= 0 {
        let dist = (-b - sqrt(discriminant)) / (2*a);
        if dist > 0 {
            let normal = normalize(at(r, dist));
            return hit(at(x, dist), (transform * vec4f(normal, 0)).xyz, dist);
        }
    }
    return hit(vec3f(0), vec3f(0), 0);
}
fn intersect_disk(x: ray, transform: mat4x4f, inv_transform: mat4x4f) -> hit {
    var r = x;
    transform_ray(&r, inv_transform);
    
    if (r.dir.z != 0) {
        let t = -r.orig.z / r.dir.z; 

        if t < 0 {
            return hit(vec3f(0), vec3f(0), 0);
        }
        let pos = at(r, t);
        if dot(pos, pos) > 1 {
            return hit(vec3f(0), vec3f(0), 0);
        }
        let side = sign(r.dir.z) == 1;
        if side {
            return hit(at(x, t), (transform * vec4f(0, 0, -1, 0)).xyz, t);
        }
        return hit(at(x, t), (transform * vec4f(0, 0, 1, 0)).xyz, t);
    }

    return hit(vec3f(0), vec3f(0), 0);
}
fn intersect_portal_pair(x: ray, portal_pair: portal_pair) -> portal_hit {
    var r_a = x;
    transform_ray(&r_a, portal_pair.inv_transform_a);
    
    var did_hit = false;
    var hit: portal_hit;
    if (r_a.dir.z != 0) {
        let t = -r_a.orig.z / r_a.dir.z; 

        if t > 1e-5 {
            let pos = at(r_a, t);
            if dot(pos, pos) <= 1 {
                hit = portal_hit(at(x, t), (pos.xy + vec2f(1)) / 2.0, t, portal_pair.transform_b * portal_pair.inv_transform_a);
                did_hit = true;
            }
        }
    }

    var r_b = x;
    transform_ray(&r_b, portal_pair.inv_transform_b);
    
    if (r_b.dir.z != 0) {
        let t = -r_b.orig.z / r_b.dir.z;
        
        if t > 1e-5 {
            let pos = at(r_b, t);
            if dot(pos, pos) <= 1 && (t < hit.t || !did_hit) {
                hit = portal_hit(at(x, t), (pos.xy + vec2f(1)) / 2.0, t, portal_pair.transform_a * portal_pair.inv_transform_b);
            }
        }
    }

    return hit;
}

fn ray_color(x: ray) -> vec3f {
    var r = x;
    for (var depth: u32 = 0; depth < 32; depth++) {
        var closest = hit(vec3f(0), vec3f(0), 3.4028234e+38); // biggest finite f32
        for (var i: u32 = 0; i < uniform.primitive_count; i++) {
            let p = primitives[i];
            switch p.ptype {
                case 1: {
                    // cube
                    let hit = intersect_box(r, p.transform, p.inv_transform);
                    if hit.t <= 0 {
                        continue;
                    }
                    if hit.t < closest.t {
                        closest = hit;
                    }
                }
                case 2: {
                    // sphere
                    let hit = intersect_sphere(r, p.transform, p.inv_transform);
                    if hit.t <= 0 {
                        continue;
                    }
                    if hit.t < closest.t {
                        closest = hit;
                    }
                }
                case 3: {
                    // disk
                    let hit = intersect_disk(r, p.transform, p.inv_transform);
                    if hit.t <= 0 {
                        continue;
                    }
                    if hit.t < closest.t {
                        closest = hit;
                    }
                }
                default: {
                    return vec3f(1, 0, 1);
                }
            }
        }

        var closest_portal = portal_hit(vec3f(0), vec2f(0), 3.4028234e+38, mat4x4f(vec4f(0), vec4f(0), vec4f(0), vec4f(0)));
        for (var i: u32 = 0; i < uniform.portal_pair_count; i++) {
            let p = portals[i];
            let hit = intersect_portal_pair(r, p);
            if hit.t <= 0 {
                continue;
            }
            if hit.t < closest_portal.t {
                closest_portal = hit;
            }
        }

        if closest.t == 3.4028234e+38 && closest_portal.t == 3.4028234e+38 {
            return sky_color(normalize(r.dir));
        }
        if closest.t < closest_portal.t {
            return vec3f(dot(closest.normal, -normalize(vec3f(-1, -3, 0)))) / 4 + vec3f(0.75);
        }

        if length(closest_portal.uv * 2.0 - vec2f(1.0)) > 0.98 {
            return vec3f(0);
        }

        r.orig = closest_portal.pos;
        transform_ray(&r, closest_portal.to_other);
    }
    return vec3f(1, 0, 0);
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
