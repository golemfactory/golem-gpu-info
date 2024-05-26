use super::{bytes_to_gib, GpuDetectionError, Result};
use crate::model::{Device, DeviceClocks, DeviceMemory, GpuApiInfo};
use crate::platform::{Detection, Flags, Platform};
use rocm_smi_lib::error::RocmErr;
use rocm_smi_lib::queries::performance::RsmiClkType;
use rocm_smi_lib::RocmSmi;
use std::fmt::{Debug, Display, Formatter};
use std::sync::Mutex;
use thiserror::Error;

#[derive(Error, Debug)]
pub struct AmdError(RocmErr);

impl From<RocmErr> for GpuDetectionError {
    fn from(value: RocmErr) -> Self {
        GpuDetectionError::AmdError(AmdError(value))
    }
}

impl Display for AmdError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

struct AmdPlatform;

impl Platform for AmdPlatform {
    fn name(&self) -> &str {
        "amd"
    }

    fn init(&self, _flags: Flags) -> crate::Result<Box<dyn Detection>> {
        eprintln!("try");
        let smi = Mutex::new(RocmSmi::init().inspect_err(|e| eprintln!("err={}", e.to_string()))?);
        Ok(Box::new(AmdDetector { smi }))
    }
}

struct AmdDetector {
    smi: Mutex<RocmSmi>,
}

impl Detection for AmdDetector {
    fn detect_api(&self, _api: &mut GpuApiInfo) -> crate::Result<()> {
        Ok(())
    }

    fn devices(&self) -> crate::Result<Vec<Device>> {
        let mut smi = self.smi.lock().unwrap();
        let device_count = smi.get_device_count();
        (0..device_count)
            .map(|dv_ind| device_info(&mut smi, dv_ind))
            .collect()
    }

    fn device_by_uuid(&self, uuid: &str) -> crate::Result<Option<Device>> {
        let mut smi = self.smi.lock().unwrap();
        let device_count = smi.get_device_count();
        Ok(
            if let Some((_, dv_ind)) = (0..device_count)
                .filter_map(|dv_ind| Some((smi.get_device_pcie_data(dv_ind).ok()?, dv_ind)))
                .map(|(pci, dv_ind)| (format!("{:016x}", pci.id), dv_ind))
                .find(|(id, _)| id == uuid)
            {
                Some(device_info(&mut smi, dv_ind)?)
            } else {
                None
            },
        )
    }
}

fn device_info(smi: &mut RocmSmi, dv_ind: u32) -> Result<Device> {
    let clocks = clocks(smi, dv_ind)?;
    let memory = memory(smi, dv_ind)?;
    let ids = smi.get_device_identifiers(dv_ind)?;

    Ok(Device {
        model: ids.name?,
        cuda: None,
        clocks,
        memory,
        quantity: 1,
    })
}

fn clocks(smi: &mut RocmSmi, dv_ind: u32) -> Result<DeviceClocks> {
    let sm_mhz = smi
        .get_device_frequency(dv_ind, RsmiClkType::RsmiClkTypeSys)?
        .supported
        .into_iter()
        .filter_map(|x| x.try_into().ok())
        .max()
        .unwrap_or_default();
    let memory_mhz = smi
        .get_device_frequency(dv_ind, RsmiClkType::RsmiClkTypeMem)?
        .supported
        .into_iter()
        .filter_map(|x| x.try_into().ok())
        .max()
        .unwrap_or_default();
    let graphics_mhz = smi
        .get_device_frequency(dv_ind, RsmiClkType::RsmiClkTypeDcef)?
        .supported
        .into_iter()
        .filter_map(|x| x.try_into().ok())
        .max()
        .unwrap_or_default();

    Ok(DeviceClocks {
        graphics_mhz,
        memory_mhz,
        sm_mhz,
        video_mhz: None,
    })
}

fn memory(smi: &mut RocmSmi, dv_ind: u32) -> Result<DeviceMemory> {
    let mem = smi.get_device_memory_data(dv_ind)?;
    let total_gib = bytes_to_gib(mem.vram_total);

    Ok(DeviceMemory {
        bandwidth_gib: None,
        total_gib,
    })
}

static AMD_PLATFORM: AmdPlatform = AmdPlatform;

pub fn platform() -> &'static dyn Platform {
    &AMD_PLATFORM
}
