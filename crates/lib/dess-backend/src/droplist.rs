use ash::vk;
use gpu_alloc_ash::AshMemoryDevice;
use gpu_descriptor_ash::AshDescriptorDevice;

use crate::{DescriptorAllocator, DescriptorSet, GpuMemory, GpuMemoryAllocator};

#[derive(Debug, Default)]
pub struct DropList {
    images: Vec<vk::Image>,
    buffers: Vec<vk::Buffer>,
    memory: Vec<GpuMemory>,
    descriptors: Vec<DescriptorSet>,
}

impl DropList {
    pub fn drop_image(&mut self, image: vk::Image) {
        self.images.push(image);
    }

    pub fn drop_buffer(&mut self, buffer: vk::Buffer) {
        self.buffers.push(buffer);
    }

    pub fn drop_descriptor_set(&mut self, ds: DescriptorSet) {
        self.descriptors.push(ds);
    }

    pub fn drop_memory(&mut self, memory: GpuMemory) {
        self.memory.push(memory);
    }

    pub fn cleanup(
        &mut self,
        device: &ash::Device,
        memory_allocator: &mut GpuMemoryAllocator,
        descriptor_allocator: &mut DescriptorAllocator,
    ) {
        self.images.drain(..).for_each(|image| unsafe {
            device.destroy_image(image, None);
        });
        self.buffers.drain(..).for_each(|buffer| unsafe {
            device.destroy_buffer(buffer, None);
        });
        self.memory.drain(..).for_each(|memory| unsafe {
            memory_allocator.dealloc(AshMemoryDevice::wrap(device), memory)
        });
        unsafe {
            descriptor_allocator.free(
                AshDescriptorDevice::wrap(device),
                self.descriptors.drain(..),
            )
        };
    }
}
