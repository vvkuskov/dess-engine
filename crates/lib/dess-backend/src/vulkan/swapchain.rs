use std::{
    slice,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use ash::vk;
use log::info;

use crate::{
    BackendError,
    vulkan::{Device, Image, ImageDesc, Surface},
};

#[derive(Debug)]
struct SwapchainImage {
    pub image: Image,
    pub acquire: vk::Semaphore,
    pub render_finished: vk::Semaphore,
}

pub struct AcquiredImage<'a> {
    swapchain: &'a Swapchain,
    image: &'a SwapchainImage,
    index: u32,
}

impl<'a> AcquiredImage<'a> {
    pub fn present(self) -> Result<PresentResult, BackendError> {
        self.swapchain.present(self.image, self.index)
    }

    pub fn image(&self) -> &Image {
        &self.image.image
    }

    pub fn render_finished(&self) -> vk::Semaphore {
        self.image.render_finished
    }

    pub fn acquired(&self) -> vk::Semaphore {
        self.image.acquire
    }
}

impl SwapchainImage {
    fn new(device: &ash::Device, image: Image) -> Result<Self, BackendError> {
        let acquire =
            unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }?;
        let render_finished =
            unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }?;
        Ok(Self {
            image,
            acquire,
            render_finished,
        })
    }

    fn free(self, device: &ash::Device) {
        self.image.clear_views();
        unsafe {
            device.destroy_semaphore(self.acquire, None);
            device.destroy_semaphore(self.render_finished, None);
        }
    }
}

pub enum AcquireResult<'a> {
    Image(AcquiredImage<'a>),
    NeedRecreate,
}

pub enum PresentResult {
    Presented,
    NeedRecreate,
}

pub struct Swapchain {
    device: Arc<Device>,
    loader: ash::khr::swapchain::Device,
    raw: vk::SwapchainKHR,
    images: Vec<SwapchainImage>,
    current_frame: AtomicUsize,
}

impl Swapchain {
    pub fn new(device: Arc<Device>, surface: Surface) -> Result<Self, BackendError> {
        let loader = ash::khr::swapchain::Device::new(&device.instance.raw, &device.raw);
        let surface_caps = unsafe {
            surface
                .loader
                .get_physical_device_surface_capabilities(device.pdevice.raw, surface.raw)
        }?;
        let mut desired_image_count = 3.max(surface_caps.min_image_count);
        if surface_caps.max_image_count != 0 {
            desired_image_count = desired_image_count.min(surface_caps.max_image_count);
        }
        let desired_present_modes = [vk::PresentModeKHR::FIFO_RELAXED, vk::PresentModeKHR::FIFO];
        let supported_present_modes = unsafe {
            surface
                .loader
                .get_physical_device_surface_present_modes(device.pdevice.raw, surface.raw)
        }?;
        let present_mode = desired_present_modes
            .into_iter()
            .find(|mode| supported_present_modes.contains(mode))
            .unwrap_or(vk::PresentModeKHR::FIFO);
        info!(
            "Swapchain image count: {} present mode: {:?}",
            desired_image_count, present_mode
        );
        let pre_transform = if surface_caps
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_caps.current_transform
        };
        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.raw)
            .min_image_count(desired_image_count)
            .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .image_format(vk::Format::A8B8G8R8_UNORM_PACK32)
            .image_extent(surface_caps.current_extent)
            .image_usage(vk::ImageUsageFlags::STORAGE)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(pre_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .image_array_layers(1);
        let swapchain = unsafe { loader.create_swapchain(&create_info, None) }?;
        let images = unsafe { loader.get_swapchain_images(swapchain) }?;
        let images = images
            .into_iter()
            .map(|image| {
                let desc = ImageDesc {
                    ty: vk::ImageType::TYPE_2D,
                    format: vk::Format::A8B8G8R8_UNORM_PACK32,
                    usage: vk::ImageUsageFlags::STORAGE,
                    flags: vk::ImageCreateFlags::empty(),
                    extents: [
                        surface_caps.current_extent.width,
                        surface_caps.current_extent.height,
                        1,
                    ],
                    tiling: vk::ImageTiling::OPTIMAL,
                    mip_levels: 1,
                    array_elements: 1,
                };
                SwapchainImage::new(&device.raw, Image::external(device.clone(), image, desc))
                    .unwrap()
            })
            .collect();
        Ok(Swapchain {
            device,
            loader,
            raw: swapchain,
            images,
            current_frame: AtomicUsize::default(),
        })
    }

    pub fn acquire_next_image(&self) -> Result<AcquireResult<'_>, BackendError> {
        let index = self.current_frame.load(Ordering::Acquire) % self.images.len();
        let image = &self.images[index];
        let present_index = unsafe {
            self.loader
                .acquire_next_image(self.raw, u64::MAX, image.acquire, vk::Fence::null())
        };
        match present_index {
            Ok((present_index, _)) => {
                assert_eq!(present_index, index as _);
                self.current_frame.fetch_add(1, Ordering::Release);
                Ok(AcquireResult::Image(AcquiredImage {
                    swapchain: self,
                    image,
                    index: present_index,
                }))
            }
            Err(err) => {
                if err == vk::Result::ERROR_OUT_OF_DATE_KHR || err == vk::Result::SUBOPTIMAL_KHR {
                    Ok(AcquireResult::NeedRecreate)
                } else {
                    Err(BackendError::VulkanError(err))
                }
            }
        }
    }

    fn present(&self, image: &SwapchainImage, index: u32) -> Result<PresentResult, BackendError> {
        let info = vk::PresentInfoKHR::default()
            .image_indices(slice::from_ref(&index))
            .wait_semaphores(slice::from_ref(&image.render_finished))
            .swapchains(slice::from_ref(&self.raw));
        match unsafe { self.loader.queue_present(self.device.main_queue.raw, &info) } {
            Ok(_) => Ok(PresentResult::Presented),
            Err(err) => {
                if err == vk::Result::ERROR_OUT_OF_DATE_KHR || err == vk::Result::SUBOPTIMAL_KHR {
                    Ok(PresentResult::NeedRecreate)
                } else {
                    Err(BackendError::VulkanError(err))
                }
            }
        }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        self.images
            .drain(..)
            .for_each(|image| image.free(&self.device.raw));
        unsafe { self.loader.destroy_swapchain(self.raw, None) };
    }
}
