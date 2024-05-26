#![deny(missing_docs)]
#![forbid(unsafe_code)]
//! GPU Device detection and offer builder.

pub mod model;

#[cfg(feature = "amd")]
mod amd;
#[cfg(not(feature = "amd"))]
mod amd {
    #[derive(thiserror::Error, Debug)]
    #[error("AMD never")]
    pub struct AmdError {
        _inner: (),
    }
}

#[cfg(feature = "cuda")]
mod cuda;
mod platform;

use crate::model::Device;
use crate::platform::{Detection, Flags, Platform};
pub use model::Gpu;
use static_assertions::*;
use std::collections::BTreeSet;
use std::mem;
use std::result::Result as StdResult;
use thiserror::Error;

/// Errors
#[derive(Error, Debug)]
pub enum GpuDetectionError {
    /// Failed to load GPU driver.
    #[error("libloading error occurred: {0}")]
    LibloadingError(#[from] libloading::Error),

    /// Failed to attach to device
    #[error("Failed to access GPU error: {0}")]
    GpuAccessError(String),

    /// Problem with reading device properties.
    #[error("Failed to access GPU info error: {0}")]
    GpuInfoAccessError(String),

    /// Other driver errors.
    #[error("NVML error occurred: {0}")]
    Unknown(String),

    /// Required driver not found.
    #[error("Driver not found")]
    NotFound,

    /// Amd driver error
    #[error(transparent)]
    AmdError(#[from] amd::AmdError),
}

type Result<T> = StdResult<T, GpuDetectionError>;

/// Initialize device discovery backends.
pub struct GpuDetectionBuilder {
    force: BTreeSet<&'static str>,
    unstable: bool,

    platforms: Vec<&'static dyn Platform>,
}

impl Default for GpuDetectionBuilder {
    fn default() -> Self {
        let force = Default::default();
        let unstable = false;
        let platforms = vec![
            #[cfg(feature = "cuda")]
            cuda::platform(),
            #[cfg(feature = "amd")]
            amd::platform(),
        ];
        Self {
            force,
            unstable,
            platforms,
        }
    }
}

/// Device detection service.
pub struct GpuDetection {
    detections: Vec<Box<dyn Detection>>,
}

assert_impl_all!(GpuDetection: Send, Sync);

impl GpuDetectionBuilder {
    /// Queries about devices will result in an error if
    /// NVIDIA Management Library is not available in the current environment.
    pub fn force_cuda(mut self) -> Self {
        self.force.insert("cuda");
        self
    }

    /// Queries may return information about which we are not certain.
    pub fn unstable_props(mut self) -> Self {
        self.unstable = true;
        self
    }

    /// Initializes backends.
    pub fn init(mut self) -> Result<GpuDetection> {
        let detections = self
            .platforms
            .into_iter()
            .filter_map(|platform| {
                let force = self.force.remove(platform.name());
                match platform.init(Flags {
                    unstable: self.unstable,
                    force,
                }) {
                    Ok(v) => Some(Ok(v)),
                    Err(e) if force => Some(Err(e)),
                    // skip error if not forced.
                    _ => None,
                }
            })
            .collect::<Result<Vec<_>>>()?;

        if !self.force.is_empty() {
            return Err(GpuDetectionError::GpuAccessError(format!(
                "missing forced platforms: {:?}",
                self.force
            )));
        }
        Ok(GpuDetection { detections })
    }
}

impl GpuDetection {
    /// Detects all available GPUs..
    pub fn detect(&self) -> Result<Gpu> {
        let mut api = Default::default();
        let mut device = Vec::new();

        for detector in &self.detections {
            detector.detect_api(&mut api)?;

            let mut it = detector.devices()?.into_iter();
            if let Some(mut dev) = it.next() {
                for next_dev in it {
                    //while let Some(next_dev) = it.next() {
                    if next_dev.model == dev.model
                        && next_dev.cuda == dev.cuda
                        && next_dev.clocks == dev.clocks
                        && next_dev.memory == dev.memory
                    {
                        dev.quantity += 1;
                    } else {
                        device.push(mem::replace(&mut dev, next_dev));
                    }
                }
                device.push(dev);
            }
        }

        Ok(Gpu { api, device })
    }

    /// Finds single device by uuid.
    pub fn search_by_uuid(&self, uuid: &str) -> Result<Device> {
        let mut last_err = None;
        for detector in &self.detections {
            match detector.device_by_uuid(uuid) {
                Ok(Some(device)) => return Ok(device),
                Err(e) => {
                    last_err = Some(e);
                }
                _ => (),
            }
        }
        Err(last_err.unwrap_or(GpuDetectionError::NotFound))
    }
}

#[cfg(any(feature = "cuda", feature = "amd"))]
fn bytes_to_gib(memory: u64) -> f32 {
    (memory as f64 / 1024.0 / 1024.0 / 1024.0) as f32
}

#[cfg(test)]
mod test {
    use crate::model;
    use crate::model::{Device, GpuApiInfo};
    use crate::platform::{Detection, Flags, Platform};

    #[derive(Clone)]
    struct TestPlatformDetection {
        devices: Vec<Device>,
    }

    impl Platform for TestPlatformDetection {
        fn name(&self) -> &str {
            "test"
        }

        fn init(&self, _flags: Flags) -> crate::Result<Box<dyn Detection>> {
            Ok(Box::new(self.clone()))
        }
    }

    impl Detection for TestPlatformDetection {
        fn detect_api(&self, _api: &mut GpuApiInfo) -> crate::Result<()> {
            Ok(())
        }

        fn devices(&self) -> crate::Result<Vec<Device>> {
            Ok(self.devices.clone())
        }

        fn device_by_uuid(&self, _uuid: &str) -> crate::Result<Option<Device>> {
            Ok(None)
        }
    }

    #[test]
    fn test_aggregation() {
        let gpu = Device {
            model: "NVIDIA GeForce RTX 3090".to_string(),
            cuda: model::DeviceCuda {
                enabled: true,
                cores: 10496,
                caps: "8.6".to_string(),
            }
            .into(),
            clocks: model::DeviceClocks {
                graphics_mhz: 2100,
                memory_mhz: 9751,
                sm_mhz: 2100,
                video_mhz: 1950.into(),
            },
            memory: model::DeviceMemory {
                bandwidth_gib: 936.into(),
                total_gib: 24.0,
            },
            quantity: 1,
        };
        let platform: Box<dyn Platform> = Box::new(TestPlatformDetection {
            devices: vec![gpu.clone(), gpu.clone()],
        });

        let mut b = super::GpuDetectionBuilder::default();
        b.platforms = vec![Box::leak(platform)];
        let gpu = b
            .init()
            .expect("failed to initialize")
            .detect()
            .expect("mock detection");

        assert_eq!(gpu.device.len(), 1);
        let dev = gpu.device.first().unwrap();
        assert_eq!(dev.quantity, 2);
    }
}
