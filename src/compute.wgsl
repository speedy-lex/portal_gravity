@group(0) @binding(0) var out: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let pos = id.xy;
    let size = textureDimensions(out);
    if (pos.x >= size.x || pos.y >= size.y) {
        return;
    }
    let uv = vec2<f32>(pos) / vec2<f32>(size);
    textureStore(out, pos, vec4<f32>(uv, 0.0, 1.0));
}
