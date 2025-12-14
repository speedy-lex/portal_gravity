@group(0) @binding(0) var out: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) index: u32) {
    let pos = id.xy;
    textureStore(out, pos, vec4<f32>(1.0, 0.0, 0.0, 1.0));
}
