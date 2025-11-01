use std::{collections::HashMap, fmt::Debug, mem, sync::Arc};

use ash::vk;
use gpu_alloc_ash::AshMemoryDevice;
use gpu_descriptor::{DescriptorSetLayoutCreateFlags, DescriptorTotalCount};
use gpu_descriptor_ash::AshDescriptorDevice;
use log::info;
use parking_lot::Mutex;

use crate::{
    BackendError, DescriptorAllocator, DescriptorSet, GpuMemory, GpuMemoryAllocator,
    droplist::DropList,
    vulkan::{Instance, PhysicalDevice},
};

#[derive(Debug, Clone, Copy)]
struct Queue {
    raw: vk::Queue,
    pub queue_family_index: u32,
}

impl Queue {
    fn new(raw: vk::Queue, queue_family_index: u32) -> Self {
        Self {
            raw,
            queue_family_index,
        }
    }
}

pub struct Device {
    pub raw: ash::Device,
    pdevice: PhysicalDevice,
    instance: Arc<Instance>,
    main_queue: Queue,
    current_drop_list: Mutex<DropList>,
    memory_allocator: Mutex<GpuMemoryAllocator>,
    descriptor_allocator: Mutex<DescriptorAllocator>,
    frames: [Mutex<Arc<DeviceFrame>>; 2],
    samplers: HashMap<SamplerDesc, vk::Sampler>,
}

impl Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("raw", &self.raw.handle())
            .field("pdevice", &self.pdevice)
            .field("instance", &self.instance)
            .field("main_queue", &self.main_queue)
            .finish()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CommandBuffer {
    cb: vk::CommandBuffer,
    fence: vk::Fence,
}

impl CommandBuffer {
    fn new(device: &ash::Device, pool: vk::CommandPool) -> Result<Self, BackendError> {
        let cb_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY);
        let cb = unsafe { device.allocate_command_buffers(&cb_info) }?[0];
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let fence = unsafe { device.create_fence(&fence_info, None) }?;
        Ok(Self { cb, fence })
    }

    pub fn free(&self, device: &ash::Device) {
        unsafe { device.destroy_fence(self.fence, None) };
    }
}

#[derive(Debug)]
struct DeviceFrame {
    pool: vk::CommandPool,
    pub swapchain_acquired: vk::Semaphore,
    pub rendering_finished: vk::Semaphore,
    drop_list: Mutex<DropList>,
    pub main_cb: CommandBuffer,
    pub presentation_cb: CommandBuffer,
}

pub struct Frame<'a> {
    device: &'a Device,
    queue: Queue,
    frame: Arc<DeviceFrame>,
}

impl<'a> Frame<'a> {
    pub fn submit(
        &self,
        device: &ash::Device,
        cb: CommandBuffer,
        signal: vk::Semaphore,
        signal_stage: vk::PipelineStageFlags2,
        wait: vk::Semaphore,
        wait_stage: vk::PipelineStageFlags2,
    ) -> Result<(), BackendError> {
        let command_buffer = [vk::CommandBufferSubmitInfo::default().command_buffer(cb.cb)];
        let wait = [vk::SemaphoreSubmitInfo::default()
            .semaphore(wait)
            .stage_mask(wait_stage)];
        let signal = [vk::SemaphoreSubmitInfo::default()
            .semaphore(signal)
            .stage_mask(signal_stage)];
        let info = vk::SubmitInfo2::default()
            .command_buffer_infos(&command_buffer)
            .wait_semaphore_infos(&wait)
            .signal_semaphore_infos(&signal);
        unsafe { device.queue_submit2(self.queue.raw, &[info], cb.fence) }?;
        Ok(())
    }

    pub fn end(self) {
        self.device.end_frame(self.frame);
    }
}

impl DeviceFrame {
    fn new(device: &ash::Device, queue_family_index: u32) -> Result<Self, BackendError> {
        let pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_index);
        let pool = unsafe { device.create_command_pool(&pool_info, None) }?;
        let swapchain_acquired =
            unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }?;
        let rendering_finished =
            unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }?;
        let main_cb = CommandBuffer::new(device, pool)?;
        let presentation_cb = CommandBuffer::new(device, pool)?;
        Ok(Self {
            pool,
            swapchain_acquired,
            rendering_finished,
            main_cb,
            presentation_cb,
            drop_list: DropList::default().into(),
        })
    }

    fn reset(
        &self,
        device: &ash::Device,
        memory_allocator: &mut GpuMemoryAllocator,
        descriptor_allocator: &mut DescriptorAllocator,
    ) -> Result<(), BackendError> {
        unsafe { device.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty()) }?;
        self.drop_list
            .lock()
            .cleanup(device, memory_allocator, descriptor_allocator);
        Ok(())
    }

    fn free(&self, device: &ash::Device) {
        unsafe { device.destroy_command_pool(self.pool, None) };
        self.main_cb.free(device);
        self.presentation_cb.free(device);
        unsafe {
            device.destroy_semaphore(self.rendering_finished, None);
            device.destroy_semaphore(self.swapchain_acquired, None);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SamplerDesc(vk::Filter, vk::SamplerMipmapMode, vk::SamplerAddressMode);

impl Device {
    pub fn new(
        instance: Arc<Instance>,
        pdevice: PhysicalDevice,
    ) -> Result<Arc<Device>, BackendError> {
        let mut syncronization2 =
            vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        let mut maintenance4 = vk::PhysicalDeviceMaintenance4Features::default().maintenance4(true);
        let mut buffer_device_address =
            vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true);
        let mut dynamic_rendering =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        let mut descriptor_indexing = vk::PhysicalDeviceDescriptorIndexingFeatures::default()
            .runtime_descriptor_array(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_storage_buffer_update_after_bind(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .shader_storage_buffer_array_non_uniform_indexing(true);

        let main_queue = pdevice
            .queue_families
            .iter()
            .filter(|queue| {
                queue
                    .properties
                    .queue_flags
                    .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
            })
            .copied()
            .next()
            .ok_or(BackendError::NoSuitableQueue)?;
        let queue_priorities = [1.0];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(main_queue.index)
            .queue_priorities(&queue_priorities)];

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .push_next(&mut syncronization2)
            .push_next(&mut maintenance4)
            .push_next(&mut buffer_device_address)
            .push_next(&mut dynamic_rendering)
            .push_next(&mut descriptor_indexing);
        let device = unsafe {
            instance
                .raw
                .create_device(pdevice.raw, &device_create_info, None)
        }?;
        info!("Created a vulkan device");
        let main_queue = Queue::new(
            unsafe { device.get_device_queue(main_queue.index, 0) },
            main_queue.index,
        );
        let frame1 = Mutex::new(Arc::new(DeviceFrame::new(
            &device,
            main_queue.queue_family_index,
        )?));
        let frame2 = Mutex::new(Arc::new(DeviceFrame::new(
            &device,
            main_queue.queue_family_index,
        )?));
        let memory_allocator = Mutex::new(GpuMemoryAllocator::new(
            gpu_alloc::Config {
                dedicated_threshold: 32 * 1024 * 1024,
                preferred_dedicated_threshold: 16 * 1024 * 1024,
                transient_dedicated_threshold: 16 * 1024 * 1024,
                starting_free_list_chunk: 256 * 1024 * 1024,
                final_free_list_chunk: 8 * 1024 * 1024,
                minimal_buddy_size: 64 * 1024,
                initial_buddy_dedicated_size: 64 * 1024 * 1024,
            },
            unsafe {
                gpu_alloc_ash::device_properties(&instance.raw, instance.version(), pdevice.raw)
            }?,
        ));
        let descriptor_allocator = Mutex::new(DescriptorAllocator::new(2));
        let samplers = Device::create_samplers(&device)?;
        Ok(Self {
            raw: device,
            pdevice,
            instance,
            main_queue,
            frames: [frame1, frame2],
            current_drop_list: DropList::default().into(),
            memory_allocator,
            descriptor_allocator,
            samplers,
        }
        .into())
    }

    pub fn allocate_memory(&self, request: gpu_alloc::Request) -> Result<GpuMemory, BackendError> {
        unsafe {
            self.memory_allocator
                .lock()
                .alloc(AshMemoryDevice::wrap(&self.raw), request)
        }
        .map_err(|err| BackendError::MemoryAllocationFailed(err, request))
    }

    pub fn allocate_descriptors(
        &self,
        layout: vk::DescriptorSetLayout,
        layout_count: DescriptorTotalCount,
        count: u32,
        bindless: bool,
    ) -> Result<Vec<DescriptorSet>, BackendError> {
        let flags = if bindless {
            DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND
        } else {
            DescriptorSetLayoutCreateFlags::empty()
        };
        let descriptors = unsafe {
            self.descriptor_allocator.lock().allocate(
                AshDescriptorDevice::wrap(&self.raw),
                &layout,
                flags,
                &layout_count,
                count,
            )
        }?;
        Ok(descriptors)
    }

    pub fn with_drop_list<CB: FnOnce(&mut DropList)>(&self, cb: CB) {
        cb(&mut self.current_drop_list.lock());
    }

    fn begin_frame(&self) -> Result<Arc<DeviceFrame>, BackendError> {
        let mut frame = self.frames[0].lock();
        {
            let frame =
                Arc::get_mut(&mut frame).expect("Frame is used by something, can't start new");
            unsafe {
                self.raw.wait_for_fences(
                    &[frame.main_cb.fence, frame.presentation_cb.fence],
                    true,
                    u64::MAX,
                )?;
            }
            frame.reset(
                &self.raw,
                &mut self.memory_allocator.lock(),
                &mut self.descriptor_allocator.lock(),
            )?;
            let mut frame_drop_list = frame.drop_list.lock();
            let mut current_drop_list = self.current_drop_list.lock();
            mem::swap(&mut frame_drop_list, &mut current_drop_list);
        }
        Ok(frame.clone())
    }

    fn end_frame(&self, frame: Arc<DeviceFrame>) {
        drop(frame);
        let mut frame = self.frames[0].lock();
        let mut frame0 =
            Arc::get_mut(&mut frame).expect("Can't finish frame - it still hel by something");
        {
            let mut frame1 = self.frames[1].lock();
            let mut frame1 = Arc::get_mut(&mut frame1).unwrap();
            mem::swap(&mut frame0, &mut frame1);
        }
    }

    pub fn frame<'a>(&'a self) -> Result<Frame<'a>, BackendError> {
        let frame = self.begin_frame()?;
        Ok(Frame {
            device: self,
            queue: self.main_queue,
            frame,
        })
    }

    pub fn get_sampler(&self, desc: SamplerDesc) -> Option<vk::Sampler> {
        self.samplers.get(&desc).copied()
    }

    fn create_samplers(
        device: &ash::Device,
    ) -> Result<HashMap<SamplerDesc, vk::Sampler>, BackendError> {
        let texel_filters = [vk::Filter::NEAREST, vk::Filter::LINEAR];
        let mipmap_modes = [
            vk::SamplerMipmapMode::NEAREST,
            vk::SamplerMipmapMode::LINEAR,
        ];
        let address_modes = [
            vk::SamplerAddressMode::REPEAT,
            vk::SamplerAddressMode::CLAMP_TO_EDGE,
        ];
        let mut samplers = HashMap::new();
        for filter in texel_filters {
            for mipmap_mode in mipmap_modes {
                for address_mode in address_modes {
                    let anisotropy = filter == vk::Filter::LINEAR;
                    let info = vk::SamplerCreateInfo::default()
                        .mag_filter(filter)
                        .min_filter(filter)
                        .mipmap_mode(mipmap_mode)
                        .address_mode_u(address_mode)
                        .address_mode_v(address_mode)
                        .address_mode_w(address_mode)
                        .max_lod(vk::LOD_CLAMP_NONE)
                        .max_anisotropy(16.0)
                        .anisotropy_enable(anisotropy);
                    let sampler = unsafe { device.create_sampler(&info, None) }?;
                    samplers.insert(SamplerDesc(filter, mipmap_mode, address_mode), sampler);
                }
            }
        }
        Ok(samplers)
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { self.raw.device_wait_idle() }.unwrap();
        let mut memory_allocator = self.memory_allocator.lock();
        let mut descriptor_allocator = self.descriptor_allocator.lock();
        self.current_drop_list.lock().cleanup(
            &self.raw,
            &mut memory_allocator,
            &mut descriptor_allocator,
        );
        for frame in &self.frames {
            let frame = frame.lock();
            frame
                .reset(&self.raw, &mut memory_allocator, &mut descriptor_allocator)
                .unwrap();
            frame.free(&self.raw);
        }
        unsafe {
            memory_allocator.cleanup(AshMemoryDevice::wrap(&self.raw));
            descriptor_allocator.cleanup(AshDescriptorDevice::wrap(&self.raw));
        }
        unsafe { self.raw.destroy_device(None) };
    }
}
