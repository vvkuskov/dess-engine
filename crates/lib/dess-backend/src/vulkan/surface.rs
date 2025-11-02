use std::fmt::Debug;

use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::{BackendError, vulkan::Instance};

pub struct Surface {
    pub raw: vk::SurfaceKHR,
    pub loader: ash::khr::surface::Instance,
}

impl Debug for Surface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Surface").field("raw", &self.raw).finish()
    }
}

impl Instance {
    pub fn create_surface(
        &self,
        display: &impl HasDisplayHandle,
        window: &impl HasWindowHandle,
    ) -> Result<Surface, BackendError> {
        let surface = unsafe {
            ash_window::create_surface(
                &self.entry,
                &self.raw,
                display.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )
        }?;
        let loader = ash::khr::surface::Instance::new(&self.entry, &self.raw);
        Ok(Surface {
            raw: surface,
            loader,
        })
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_surface(self.raw, None) };
    }
}
