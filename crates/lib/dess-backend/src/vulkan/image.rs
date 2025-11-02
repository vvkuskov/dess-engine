use std::{collections::HashMap, sync::Arc};

use ash::vk;
use log::info;
use parking_lot::Mutex;

use crate::{BackendError, DropList, GpuMemory, vulkan::Device};

#[derive(Debug, Clone, Copy)]
pub struct ImageDesc {
    pub ty: vk::ImageType,
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
    pub flags: vk::ImageCreateFlags,
    pub extents: [u32; 3],
    pub tiling: vk::ImageTiling,
    pub mip_levels: u32,
    pub array_elements: u32,
}

fn mip_count(extent: u32) -> u32 {
    32 - extent.leading_zeros()
}

impl ImageDesc {
    pub fn new(format: vk::Format, ty: vk::ImageType, extents: [u32; 3]) -> Self {
        Self {
            ty,
            format,
            usage: vk::ImageUsageFlags::default(),
            flags: vk::ImageCreateFlags::empty(),
            extents,
            tiling: vk::ImageTiling::OPTIMAL,
            mip_levels: 1,
            array_elements: 1,
        }
    }

    pub fn cube(format: vk::Format, size: u32) -> Self {
        Self {
            ty: vk::ImageType::TYPE_2D,
            format,
            usage: vk::ImageUsageFlags::default(),
            flags: vk::ImageCreateFlags::CUBE_COMPATIBLE,
            extents: [size, size, 1],
            tiling: vk::ImageTiling::OPTIMAL,
            mip_levels: 1,
            array_elements: 6,
        }
    }

    pub fn ty(mut self, value: vk::ImageType) -> Self {
        self.ty = value;
        self
    }

    pub fn format(mut self, value: vk::Format) -> Self {
        self.format = value;
        self
    }

    pub fn usage(mut self, value: vk::ImageUsageFlags) -> Self {
        self.usage = value;
        self
    }

    pub fn flags(mut self, value: vk::ImageCreateFlags) -> Self {
        self.flags = value;
        self
    }

    pub fn tiling(mut self, value: vk::ImageTiling) -> Self {
        self.tiling = value;
        self
    }

    pub fn mip_levels(mut self, value: u32) -> Self {
        self.mip_levels = value;
        self
    }

    pub fn array_elements(mut self, value: u32) -> Self {
        self.array_elements = value;
        self
    }

    pub fn all_mip_levels(mut self) -> Self {
        self.mip_levels = mip_count(self.extents[0])
            .max(self.extents[1])
            .max(self.extents[2]);
        self
    }

    pub fn div_extents(mut self, div: [u32; 3]) -> Self {
        for (extent, div) in self.extents.iter_mut().zip(&div) {
            *extent = (*extent / *div).max(1);
        }
        self
    }

    pub fn half_res(self) -> Self {
        self.div_extents([2, 2, 2])
    }

    pub fn extent_2d(&self) -> [u32; 2] {
        [self.extents[0], self.extents[1]]
    }

    fn get_create_info(&self) -> vk::ImageCreateInfo<'_> {
        vk::ImageCreateInfo::default()
            .format(self.format)
            .extent(vk::Extent3D {
                width: self.extents[0],
                height: self.extents[1],
                depth: self.extents[2],
            })
            .flags(self.flags)
            .image_type(self.ty)
            .mip_levels(self.mip_levels)
            .array_layers(self.array_elements)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(self.tiling)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .usage(self.usage)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageViewDesc {
    pub ty: Option<vk::ImageViewType>,
    pub format: Option<vk::Format>,
    pub aspect_flags: vk::ImageAspectFlags,
    pub base_mip_level: u32,
    pub level_count: Option<u32>,
}

impl ImageViewDesc {
    pub fn new(aspect_flags: vk::ImageAspectFlags) -> Self {
        Self {
            ty: None,
            format: None,
            aspect_flags,
            base_mip_level: 0,
            level_count: None,
        }
    }

    pub fn color() -> Self {
        Self::new(vk::ImageAspectFlags::COLOR)
    }

    pub fn depth() -> Self {
        Self::new(vk::ImageAspectFlags::DEPTH)
    }

    pub fn stencil() -> Self {
        Self::new(vk::ImageAspectFlags::STENCIL)
    }

    pub fn ty(mut self, value: vk::ImageViewType) -> Self {
        self.ty = Some(value);
        self
    }

    pub fn format(mut self, value: vk::Format) -> Self {
        self.format = Some(value);
        self
    }

    pub fn aspect(mut self, value: vk::ImageAspectFlags) -> Self {
        self.aspect_flags = value;
        self
    }

    pub fn base_mip_level(mut self, value: u32) -> Self {
        self.base_mip_level = value;
        self
    }

    pub fn level_count(mut self, value: u32) -> Self {
        self.level_count = Some(value);
        self
    }

    fn get_image_view_create_info(
        &self,
        desc: ImageDesc,
        image: vk::Image,
    ) -> vk::ImageViewCreateInfo<'_> {
        vk::ImageViewCreateInfo::default()
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::G,
                b: vk::ComponentSwizzle::B,
                a: vk::ComponentSwizzle::A,
            })
            .view_type(self.get_view_type(desc))
            .format(self.format.unwrap_or(desc.format))
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(self.aspect_flags)
                    .base_mip_level(self.base_mip_level)
                    .level_count(self.level_count.unwrap_or(desc.mip_levels))
                    .base_array_layer(0)
                    .layer_count(desc.array_elements),
            )
    }

    fn get_view_type(&self, desc: ImageDesc) -> vk::ImageViewType {
        match desc.ty {
            vk::ImageType::TYPE_1D if desc.array_elements > 1 => vk::ImageViewType::TYPE_1D_ARRAY,
            vk::ImageType::TYPE_1D => vk::ImageViewType::TYPE_1D,
            vk::ImageType::TYPE_2D
                if desc.flags.contains(vk::ImageCreateFlags::CUBE_COMPATIBLE)
                    && desc.array_elements / 6 > 1 =>
            {
                vk::ImageViewType::CUBE_ARRAY
            }
            vk::ImageType::TYPE_2D
                if desc.flags.contains(vk::ImageCreateFlags::CUBE_COMPATIBLE)
                    && desc.array_elements == 6 =>
            {
                vk::ImageViewType::CUBE
            }
            vk::ImageType::TYPE_2D if desc.array_elements > 1 => vk::ImageViewType::TYPE_2D_ARRAY,
            vk::ImageType::TYPE_2D => vk::ImageViewType::TYPE_2D,
            vk::ImageType::TYPE_3D => vk::ImageViewType::TYPE_3D,
            _ => panic!("Can't create image view type from {:?}", desc),
        }
    }
}

#[derive(Debug)]
pub struct Image {
    device: Arc<Device>,
    pub raw: vk::Image,
    pub desc: ImageDesc,
    views: Mutex<HashMap<ImageViewDesc, vk::ImageView>>,
    memory: Option<GpuMemory>,
}

unsafe impl Send for Image {}
unsafe impl Sync for Image {}

impl Image {
    pub fn new(device: Arc<Device>, desc: ImageDesc) -> Result<Image, BackendError> {
        info!("Create image {:?}", desc);
        let info = desc.get_create_info();
        let image = unsafe { device.raw.create_image(&info, None) }?;
        let requirements = unsafe { device.raw.get_image_memory_requirements(image) };
        let request = gpu_alloc::Request {
            size: requirements.size.min(64536),
            align_mask: requirements.alignment,
            usage: gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
            memory_types: requirements.memory_type_bits,
        };
        let memory = device.allocate_memory(request)?;
        unsafe {
            device
                .raw
                .bind_image_memory(image, *memory.memory(), memory.offset())
        }?;
        Ok(Self {
            device,
            raw: image,
            desc,
            views: Default::default(),
            memory: Some(memory),
        })
    }

    pub fn external(device: Arc<Device>, image: vk::Image, desc: ImageDesc) -> Self {
        info!("Create external image {:?}", desc);
        Self {
            device,
            raw: image,
            desc,
            views: Default::default(),
            memory: None,
        }
    }

    pub fn get_or_create_view(&self, desc: ImageViewDesc) -> Result<vk::ImageView, BackendError> {
        let mut views = self.views.lock();
        if let Some(view) = views.get(&desc).copied() {
            Ok(view)
        } else {
            let create_info = desc.get_image_view_create_info(self.desc, self.raw);
            let view = unsafe { self.device.raw.create_image_view(&create_info, None) }?;
            views.insert(desc, view);
            Ok(view)
        }
    }

    pub fn clear_views(&self) {
        self.device
            .with_drop_list(|drop_list| self.clear_views_impl(drop_list));
    }

    fn clear_views_impl(&self, drop_list: &mut DropList) {
        self.views
            .lock()
            .drain()
            .for_each(|(_, view)| drop_list.drop_image_view(view));
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        if let Some(memory) = self.memory.take() {
            self.device.with_drop_list(|drop_list| {
                drop_list.drop_memory(memory);
                drop_list.drop_image(self.raw);
            });
        }
        self.clear_views();
    }
}
