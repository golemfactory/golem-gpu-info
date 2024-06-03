# gpu-detection

Library detects GPU info listed in [GAP-35](https://github.com/golemfactory/golem-architecture/blob/master/gaps/gap-35_gpu_pci_capability/gap-35_gpu_pci_capability.md).

It supports Nvidia GPUs only. Implementation uses [nvml-wrapper](https://crates.io/crates/nvml-wrapper) to access [NVML](https://developer.nvidia.com/nvidia-management-library-nvml).

Example attrubite set for 2xA30

```json
{
  "golem.inf.gpu.cuda.version": "12.3"
  "golem.inf.gpu.cuda.driver.version": "545.23.08",
  "golem.inf.gpu.d0.model": "NVIDIA A30",
  "golem.inf.gpu.d0.cuda.enabled": true,
  "golem.inf.gpu.d0.cuda.cores": true,
  "golem.inf.gpu.d0.cuda.caps": "8.0",
  "golem.inf.gpu.d0.clock.graphics.mhz": 1440,
  "golem.inf.gpu.d0.clock.memory.mhz": 1440,
  "golem.inf.gpu.d0.clock.sm.mhz": 1440,
  "golem.inf.gpu.d0.clock.video.mhz": 1305,
  "golem.inf.gpu.d0.memory.bandwidth.gib": 933,
  "golem.inf.gpu.d0.memory.total.gib": 24.0,
  "golem.inf.gpu.d0.quantity": 2
}
```

