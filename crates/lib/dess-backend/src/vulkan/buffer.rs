use std::{ptr::NonNull, sync::Arc};

use ash::vk;
use gpu_alloc_ash::AshMemoryDevice;

use crate::{BackendError, GpuMemory, vulkan::Device};

#[derive(Debug, Clone, Copy)]
pub struct BufferDesc {
    pub size: usize,
    pub usage: vk::BufferUsageFlags,
    pub location: gpu_alloc::UsageFlags,
    pub aligment: Option<u64>,
}

impl BufferDesc {
    pub fn new(size: usize, usage: vk::BufferUsageFlags) -> Self {
        Self {
            size,
            usage,
            location: gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
            aligment: None,
        }
    }

    pub fn storage(size: usize) -> Self {
        Self::new(size, vk::BufferUsageFlags::STORAGE_BUFFER)
    }

    pub fn vertex(size: usize) -> Self {
        Self::new(size, vk::BufferUsageFlags::VERTEX_BUFFER)
    }

    pub fn index(size: usize) -> Self {
        Self::new(size, vk::BufferUsageFlags::INDEX_BUFFER)
    }

    pub fn indirect(size: usize) -> Self {
        Self::new(size, vk::BufferUsageFlags::INDIRECT_BUFFER)
    }

    pub fn size(mut self, value: usize) -> Self {
        self.size = value;
        self
    }

    pub fn usage(mut self, value: vk::BufferUsageFlags) -> Self {
        self.usage = value;
        self
    }

    pub fn device_memory(mut self) -> Self {
        self.location = gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS;
        self
    }

    pub fn upload(mut self) -> Self {
        self.location = gpu_alloc::UsageFlags::UPLOAD;
        self
    }

    pub fn download(mut self) -> Self {
        self.location = gpu_alloc::UsageFlags::DOWNLOAD;
        self
    }
}

#[derive(Debug)]
pub struct Buffer {
    device: Arc<Device>,
    pub raw: vk::Buffer,
    pub desc: BufferDesc,
    memory: Option<GpuMemory>,
    mapping: Option<NonNull<u8>>,
}

impl Buffer {
    pub fn new(device: Arc<Device>, desc: BufferDesc) -> Result<Self, BackendError> {
        let info = vk::BufferCreateInfo::default()
            .usage(desc.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .size(desc.size as _);
        let buffer = unsafe { device.raw.create_buffer(&info, None) }?;
        let requirements = unsafe { device.raw.get_buffer_memory_requirements(buffer) };
        let request = gpu_alloc::Request {
            size: desc.size as u64,
            align_mask: desc
                .aligment
                .unwrap_or(requirements.alignment)
                .max(requirements.alignment),
            usage: desc.location,
            memory_types: requirements.memory_type_bits,
        };
        let mut memory = device.allocate_memory(request)?;
        unsafe {
            device
                .raw
                .bind_buffer_memory(buffer, *memory.memory(), memory.offset())
        }?;
        let mapping = if desc.location.contains(gpu_alloc::UsageFlags::UPLOAD)
            || desc.location.contains(gpu_alloc::UsageFlags::DOWNLOAD)
        {
            Some(unsafe { memory.map(AshMemoryDevice::wrap(&device.raw), 0, desc.size) }?)
        } else {
            None
        };
        Ok(Self {
            device,
            raw: buffer,
            desc,
            memory: Some(memory),
            mapping,
        })
    }

    pub fn mapping(&self) -> Option<NonNull<u8>> {
        self.mapping
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if let Some(memory) = self.memory.take() {
            self.device.with_drop_list(|drop_list| {
                drop_list.drop_memory(memory);
                drop_list.drop_buffer(self.raw);
            });
        }
    }
}
