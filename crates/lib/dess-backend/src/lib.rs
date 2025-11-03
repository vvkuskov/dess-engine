mod chunky_list;
mod droplist;
mod error;
mod pipeline_cache;
mod shaders;
pub mod vulkan;
pub use ash::vk;
pub use droplist::*;
pub use error::*;
pub use pipeline_cache::*;
pub use shaders::*;

pub(crate) type DescriptorSet = gpu_descriptor::DescriptorSet<vk::DescriptorSet>;
pub(crate) type DescriptorAllocator =
    gpu_descriptor::DescriptorAllocator<vk::DescriptorPool, vk::DescriptorSet>;
pub(crate) type GpuMemory = gpu_alloc::MemoryBlock<vk::DeviceMemory>;
pub(crate) type GpuMemoryAllocator = gpu_alloc::GpuAllocator<vk::DeviceMemory>;
