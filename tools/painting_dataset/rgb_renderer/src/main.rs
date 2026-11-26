use naga::{Block, Statement};
use serde_json::{Value, json};
use std::{
    error::Error,
    io::{self, Read},
    time::{Duration, Instant},
};

fn block_cost(block: &Block, module: &naga::Module, depth: usize) -> Result<usize, Box<dyn Error>> {
    if depth > 16 {
        return Err("Function nesting exceeds 16".into());
    }
    let mut cost = 0;
    for statement in block.iter() {
        cost += 1;
        cost += match statement {
            Statement::Loop { .. } => {
                return Err("Loops are not supported by the bounded paint contract".into());
            }
            Statement::Block(b) => block_cost(b, module, depth)?,
            Statement::If { accept, reject, .. } => {
                block_cost(accept, module, depth)? + block_cost(reject, module, depth)?
            }
            Statement::Switch { cases, .. } => {
                let mut n = 0;
                for case in cases {
                    n += block_cost(&case.body, module, depth)?;
                }
                n
            }
            Statement::Call { function, .. } => {
                let f = &module.functions[*function];
                f.expressions.len() + block_cost(&f.body, module, depth + 1)?
            }
            _ => 0,
        };
        if cost > 4096 {
            return Err("Expanded shader work exceeds 4096 nodes".into());
        }
    }
    Ok(cost)
}

fn run() -> Result<Value, Box<dyn Error>> {
    let started = Instant::now();
    let mut input = String::new();
    io::stdin().take(65537).read_to_string(&mut input)?;
    if input.len() > 65536 {
        return Err("Request too large".into());
    }
    let req: Value = serde_json::from_str(&input)?;
    let source = req["source"].as_str().ok_or("Missing source")?;
    if source.len() > 32768 {
        return Err("Source exceeds 32768 bytes".into());
    }
    let width = req["width"].as_u64().ok_or("Missing width")?;
    let height = req["height"].as_u64().ok_or("Missing height")?;
    if !(1..=512).contains(&width) || !(1..=512).contains(&height) {
        return Err("Dimensions must be 1..512".into());
    }
    let output = req["output"].as_str().ok_or("Missing output")?;
    let user = naga::front::wgsl::parse_str(source).map_err(|e| e.emit_to_string(source))?;
    if !user.entry_points.is_empty() || !user.global_variables.is_empty() {
        return Err("Entry points, global variables and external bindings are not allowed".into());
    }
    for (_, ty) in user.types.iter() {
        if matches!(
            ty.inner,
            naga::TypeInner::Array { .. }
                | naga::TypeInner::BindingArray { .. }
                | naga::TypeInner::Struct { .. }
        ) {
            return Err(
                "Arrays and structs are not supported by the bounded paint contract".into(),
            );
        }
    }
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&user)?;
    for (_, function) in user.functions.iter() {
        if function.expressions.len() + block_cost(&function.body, &user, 0)? > 4096 {
            return Err("Expanded shader work exceeds 4096 nodes".into());
        }
    }
    let shader = format!(
        "{source}\n@group(0) @binding(0) var<storage, read_write> rgb_output: array<vec4f>;\n@compute @workgroup_size(8,8) fn painting_kernel(@builtin(global_invocation_id) id:vec3u) {{ if(id.x >= {width}u || id.y >= {height}u) {{ return; }} let uv=(vec2f(id.xy)+vec2f(0.5))/vec2f({width}.0,{height}.0); rgb_output[id.y*{width}u+id.x]=vec4f(paint(uv),1.0); }}"
    );
    let parsed = naga::front::wgsl::parse_str(&shader).map_err(|e| e.emit_to_string(&shader))?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&parsed)?;
    let pci = std::env::var("PI_GPU_PCI_BUS_ID")
        .map_err(|_| "PI_GPU_PCI_BUS_ID must explicitly select the reserved GPU")?;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN))
        .into_iter()
        .find(|a| a.get_info().device_pci_bus_id.eq_ignore_ascii_case(&pci))
        .ok_or("Reserved PCI GPU not found")?;
    let info = adapter.get_info();
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))?;
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bounded RGB painting"),
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader_module,
        entry_point: Some("painting_kernel"),
        compilation_options: Default::default(),
        cache: None,
    });
    let size = width * height * 16;
    let pixels = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: pixels.as_entire_binding(),
        }],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((width as u32).div_ceil(8), (height as u32).div_ceil(8), 1);
    }
    encoder.copy_buffer_to_buffer(&pixels, 0, &readback, 0, size);
    queue.submit([encoder.finish()]);
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(Duration::from_secs(5)),
    })?;
    rx.recv_timeout(Duration::from_secs(5))??;
    let data = slice.get_mapped_range();
    let floats: &[f32] = bytemuck::cast_slice(&data);
    if floats.iter().any(|x| !x.is_finite()) {
        return Err("Nonfinite pixels".into());
    }
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for pixel in floats.chunks_exact(4) {
        for value in &pixel[..3] {
            rgb.push((value.clamp(0.0, 1.0) * 255.0).round() as u8);
        }
    }
    image::RgbImage::from_raw(width as u32, height as u32, rgb)
        .ok_or("Invalid pixel count")?
        .save(output)?;
    Ok(
        json!({"ok": true, "image": output, "error": null, "gpu": info.name, "pci_bus_id": info.device_pci_bus_id, "width": width, "height": height, "elapsed_seconds": started.elapsed().as_secs_f64()}),
    )
}

fn main() {
    match run() {
        Ok(value) => println!("{value}"),
        Err(error) => {
            println!(
                "{}",
                json!({"ok": false, "image": null, "error": error.to_string()})
            );
            std::process::exit(1);
        }
    }
}
