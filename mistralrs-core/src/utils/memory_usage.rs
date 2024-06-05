use candle_core::Device;
use sysinfo::System;

const KB_TO_BYTES: usize = 1024;

pub struct MemoryUsage;

impl MemoryUsage {
    pub fn get_memory_available(&self, device: &Device) -> usize {
        match device {
            Device::Cpu => {
                let mut sys = System::new_all();
                sys.refresh_cpu();
                sys.free_memory() as usize * KB_TO_BYTES
            }
            Device::Cuda(_) => {
                todo!()
            }
            Device::Metal(_) => {
                todo!()
            }
        }
    }
}
