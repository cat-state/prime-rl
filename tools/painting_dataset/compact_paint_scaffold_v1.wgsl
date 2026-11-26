// Generic abstract study: not derived from any evaluation artwork.
fn oval(uv: vec2f, center: vec2f, radius: vec2f) -> f32 {
    let d = length((uv - center) / radius);
    return 1.0 - smoothstep(0.95, 1.05, d);
}
fn rectangle(uv: vec2f, center: vec2f, half_size: vec2f) -> f32 {
    let d = abs(uv - center) - half_size;
    return 1.0 - smoothstep(-0.004, 0.004, max(d.x, d.y));
}
fn paint(uv: vec2f) -> vec3f {
    var col = mix(vec3f(0.12, 0.18, 0.28), vec3f(0.55, 0.65, 0.70), uv.y);
    col = mix(col, vec3f(0.24, 0.19, 0.15), smoothstep(0.68, 0.69, uv.y));
    col = mix(col, vec3f(0.70, 0.36, 0.18), rectangle(uv, vec2f(0.36, 0.58), vec2f(0.15, 0.12)));
    col = mix(col, vec3f(0.88, 0.76, 0.42), oval(uv, vec2f(0.67, 0.42), vec2f(0.13, 0.18)));
    col = mix(col, vec3f(0.18, 0.35, 0.28), oval(uv, vec2f(0.60, 0.70), vec2f(0.22, 0.04)));
    return clamp(col, vec3f(0.0), vec3f(1.0));
}
