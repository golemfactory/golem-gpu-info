use super::Result;
use crate::model::{Device, GpuApiInfo};

pub struct Flags {
    pub unstable: bool,
    pub force: bool,
}

pub trait Platform {
    fn name(&self) -> &str;

    fn init(&self, flags: Flags) -> Result<Box<dyn Detection>>;
}

pub trait Detection: Sync + Send {
    fn detect_api(&self, api: &mut GpuApiInfo) -> Result<()>;

    fn devices(&self) -> Result<Vec<Device>>;

    fn device_by_uuid(&self, uuid: &str) -> Result<Option<Device>>;
}
