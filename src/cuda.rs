use crate::model::{Cuda, Device as GpuDevice, DeviceClocks, DeviceCuda, DeviceMemory, GpuApiInfo};
use crate::platform::{Detection, Flags, Platform};
use crate::{bytes_to_gib, GpuDetectionError};
use nvml_wrapper::error::NvmlError;
use nvml_wrapper::{enum_wrappers::device::Clock, Device, Nvml};

pub(crate) struct CudaDetection {
    flags: Flags,
    nvml: Nvml,
}

impl Detection for CudaDetection {
    fn detect_api(&self, api: &mut GpuApiInfo) -> crate::Result<()> {
        let version = self
            .cuda_version()
            .map_err(|e| GpuDetectionError::GpuInfoAccessError(e.to_string()))?;
        let driver_version = self.nvml.sys_driver_version().ok();
        api.cuda = Some(Cuda {
            version,
            driver_version,
        });
        Ok(())
    }

    fn devices(&self) -> Result<Vec<GpuDevice>, GpuDetectionError> {
        let gpu_count = self.nvml.device_count().map_err(|err| {
            GpuDetectionError::Unknown(format!("Failed to get device count. Err {}", err))
        })?;

        (0..gpu_count)
            .map(|index| {
                self.nvml
                    .device_by_index(index)
                    .and_then(|device| device_info(device, &self.flags))
            })
            .collect::<Result<_, NvmlError>>()
            .map_err(|e| GpuDetectionError::GpuAccessError(e.to_string()))
    }

    fn device_by_uuid(&self, uuid: &str) -> super::Result<Option<GpuDevice>> {
        let device = match self.nvml.device_by_uuid(uuid) {
            Ok(device) => device,
            Err(NvmlError::NotFound) => return Ok(None),
            Err(e) => return Err(GpuDetectionError::GpuAccessError(e.to_string())),
        };

        let dev_info = device_info(device, &self.flags)
            .map_err(|e| GpuDetectionError::GpuInfoAccessError(e.to_string()))?;
        Ok(Some(dev_info))
    }
}

impl CudaDetection {
    fn cuda_version(&self) -> Result<String, NvmlError> {
        let version = self.nvml.sys_cuda_driver_version()?;
        let version_major = nvml_wrapper::cuda_driver_version_major(version);
        let version_minor = nvml_wrapper::cuda_driver_version_minor(version);
        Ok(format!("{}.{}", version_major, version_minor))
    }
}

fn device_info(dev: Device, flags: &Flags) -> Result<GpuDevice, NvmlError> {
    let model = dev.name()?;
    let cuda = Some(cuda(&dev, flags)?);
    let clocks = clocks(&dev)?;
    let memory = memory(&dev, flags)?;
    Ok(GpuDevice {
        model,
        cuda,
        clocks,
        memory,
        quantity: 1,
    })
}

fn cuda(dev: &Device, _flags: &Flags) -> Result<DeviceCuda, NvmlError> {
    let enabled = true;
    let cores = dev.num_cores()?;
    let caps = compute_capability(dev)?;
    Ok(DeviceCuda {
        enabled,
        cores,
        caps,
    })
}

fn compute_capability(dev: &Device) -> Result<String, NvmlError> {
    let capability = dev.cuda_compute_capability()?;
    Ok(format!("{}.{}", capability.major, capability.minor))
}

fn clocks(dev: &Device) -> Result<DeviceClocks, NvmlError> {
    let graphics_mhz = dev.max_clock_info(Clock::Graphics)?;
    let memory_mhz = dev.max_clock_info(Clock::Memory)?;
    let sm_mhz = dev.max_clock_info(Clock::SM)?;
    let video_mhz = Some(dev.max_clock_info(Clock::Video)?);
    Ok(DeviceClocks {
        graphics_mhz,
        memory_mhz,
        sm_mhz,
        video_mhz,
    })
}

fn memory(dev: &Device, flags: &Flags) -> Result<DeviceMemory, NvmlError> {
    let total_bytes = dev.memory_info()?.total;
    let total_gib = bytes_to_gib(total_bytes);
    let bandwidth_gib = if flags.unstable {
        bandwidth_gib(dev)?
    } else {
        None
    };

    Ok(DeviceMemory {
        bandwidth_gib,
        total_gib,
    })
}

fn bandwidth_gib(dev: &Device) -> Result<Option<u32>, NvmlError> {
    let memory_bus_width = dev.memory_bus_width()?;
    let max_memory_clock = dev.max_clock_info(Clock::Memory)?;

    // `nvml` does not provide `memTransferRatemax` like `nvidia-settings` tool does.
    // Transfer rate is a result of memory clock, bus width,
    // and memory specific multiplier (for DDR it is 2)
    let data_rate = 2; // value for DDR
    let bandwidth_gib = max_memory_clock * memory_bus_width * data_rate / (1000 * 8);

    Ok(Some(bandwidth_gib))
}

struct CudaPlatform;

impl Platform for CudaPlatform {
    fn name(&self) -> &str {
        "cuda"
    }

    fn init(&self, flags: Flags) -> crate::Result<Box<dyn Detection>> {
        let nvml = match nvml_init() {
            Ok(nvlm) => nvlm,
            Err(NvmlError::LibloadingError(e)) => {
                return if flags.force {
                    Err(GpuDetectionError::LibloadingError(e))
                } else {
                    Err(GpuDetectionError::NotFound)
                }
            }
            Err(e) => return Err(GpuDetectionError::Unknown(e.to_string())),
        };
        Ok(Box::new(CudaDetection { nvml, flags }))
    }
}

// On systems without a full development environment there may not
// be `libnvidia-ml.so`. Because there is a convention to name `lib<name>.so.<version>` files
// as runtime lib.
#[cfg(target_os = "linux")]
fn nvml_init() -> std::result::Result<Nvml, NvmlError> {
    match Nvml::init() {
        Err(NvmlError::LibraryNotFound) => Nvml::builder()
            .lib_path("libnvidia-ml.so.1".as_ref())
            .init(),
        r => r,
    }
}

// on windows default `libnvidia-ml.dll` is ok.
#[cfg(not(target_os = "linux"))]
fn nvml_init() -> std::result::Result<Nvml, NvmlError> {
    Nvml::init()
}

static CUDA_PLATFORM: CudaPlatform = CudaPlatform;

pub fn platform() -> &'static dyn crate::platform::Platform {
    &CUDA_PLATFORM
}
