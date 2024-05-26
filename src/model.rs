//! GPU Device offer spec.
//!
//! This module provides structures to define basic information about
//! provider GPUs.

use serde::Serialize;

/// General information about all gpus.
#[derive(Clone, Debug, Serialize, Default)]
pub struct Gpu {
    /// Available SDKs & device drivers.
    #[serde(flatten)]
    pub api: GpuApiInfo,
    /// Lists of devices.
    pub device: Vec<Device>,
}

/// Available SDKs & device drivers.
#[derive(Clone, Debug, Serialize, Default)]
pub struct GpuApiInfo {
    /// Optional information about installed CUDA API & Drivers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cuda: Option<Cuda>,
}

/// information about installed CUDA.
#[derive(Clone, Debug, Serialize)]
pub struct Cuda {
    /// CUDA version
    pub version: String,
    /// Installed driver version.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub driver_version: Option<String>,
}

/// GPU device group information.
///
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct Device {
    /// Name of this device.
    ///
    /// alphanumeric string that denotes a particular product, e.g. Tesla C2070
    pub model: String,

    /// CUDA specific attributes for this device
    pub cuda: Option<DeviceCuda>,
    /// Device clocks.
    #[serde(rename = "clock")]
    pub clocks: DeviceClocks,
    /// Memory information.
    pub memory: DeviceMemory,

    /// Number of cards.
    pub quantity: usize,
}

/// CUDA specific attributes for single device
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct DeviceCuda {
    /// should be true if given device is supported.
    pub enabled: bool,
    /// Core count for this device.
    /// The cores represented in the count here are commonly referred to as "CUDA core
    pub cores: u32,
    /// CUDA compute capability of this Device
    pub caps: String,
}

/// Device clocks.
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct DeviceClocks {
    /// Graphics clock in MHz.
    ///
    /// For AMD: RSMI_CLK_TYPE_DCEF (Display Controller Engine Clock)
    /// For nVidia: NVML_CLOCK_GRAPHICS (Graphics clock domain)
    #[serde(rename(serialize = "graphics.mhz"))]
    pub graphics_mhz: u32,
    /// Memory clock in MHz.
    #[serde(rename(serialize = "memory.mhz"))]
    pub memory_mhz: u32,
    /// SM clock
    ///
    /// nVidia: NVML_CLOCK_SM (Streaming Multiprocessor)
    /// AMD: RSMI_FREQ_TYPE_SYS (
    #[serde(rename(serialize = "sm.mhz"))]
    pub sm_mhz: u32,
    /// Video encoder/decoder clock
    ///
    /// nVidia: NVML_CLOCK_VIDEO
    #[serde(rename(serialize = "video.mhz"))]
    pub video_mhz: Option<u32>,
}

/// Memory.
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct DeviceMemory {
    /// Peak Memory Bandwidth.
    ///
    /// unstable option.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename(serialize = "bandwidth.gib"))]
    pub bandwidth_gib: Option<u32>,
    /// Total physical device memory on device in GiB,
    #[serde(rename(serialize = "total.gib"))]
    pub total_gib: f32,
}
