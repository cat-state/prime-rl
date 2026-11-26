// CPU-only syntax/type validation; does not create a renderer or GPU device.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("expected WGSL path")?;
    let source = std::fs::read_to_string(path)?;
    let module = naga::front::wgsl::parse_str(&source)?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)?;
    println!("WGSL syntax and types valid; no GPU render performed");
    Ok(())
}
