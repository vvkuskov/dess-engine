use ash::vk;

use crate::{BackendError, vulkan::Instance};

#[derive(Debug, Clone, Copy)]
pub struct QueueFamily {
    pub index: u32,
    pub properties: vk::QueueFamilyProperties,
}

#[derive(Debug)]
pub struct PhysicalDevice {
    pub raw: vk::PhysicalDevice,
    pub queue_families: Vec<QueueFamily>,
    pub properties: vk::PhysicalDeviceProperties,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
}

impl Instance {
    pub fn get_physical_devices(&self) -> Result<Vec<PhysicalDevice>, BackendError> {
        let pdevices = unsafe { self.raw.enumerate_physical_devices() }?
            .into_iter()
            .map(|pdevice| {
                let properties = unsafe { self.raw.get_physical_device_properties(pdevice) };
                let memory_properties =
                    unsafe { self.raw.get_physical_device_memory_properties(pdevice) };
                let queue_families = unsafe {
                    self.raw
                        .get_physical_device_queue_family_properties(pdevice)
                }
                .into_iter()
                .enumerate()
                .map(|(index, properties)| QueueFamily {
                    index: index as _,
                    properties,
                })
                .collect();
                PhysicalDevice {
                    raw: pdevice,
                    queue_families,
                    properties,
                    memory_properties,
                }
            })
            .collect();
        Ok(pdevices)
    }
}
