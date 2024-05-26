use golem_gpu_info::GpuDetectionBuilder;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let detection = GpuDetectionBuilder::default().unstable_props().init()?;

    let gpu = detection.detect()?;

    serde_json::to_writer_pretty(&mut std::io::stdout(), &gpu)?;
    Ok(())
}
