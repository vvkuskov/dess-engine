use std::{
    ffi::{CString, c_void},
    fmt::Debug,
    sync::Arc,
};

use ash::vk;
use log::{error, info, trace, warn};

use crate::BackendError;

pub struct Instance {
    pub(crate) entry: ash::Entry,
    pub raw: ash::Instance,
    debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
}

impl Debug for Instance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Instance")
            .field("raw", &self.raw.handle())
            .finish()
    }
}

#[derive(Debug, Default)]
pub struct InstanceBuilder<'a> {
    debug: bool,
    title: Option<&'a str>,
}

impl<'a> InstanceBuilder<'a> {
    pub fn debug(mut self, value: bool) -> Self {
        self.debug = value;
        self
    }

    pub fn title(mut self, value: &'a str) -> Self {
        self.title = Some(value);
        self
    }

    fn extensions(&self) -> Vec<*const i8> {
        let mut names = vec![vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME.as_ptr()];
        if self.debug {
            names.push(vk::EXT_DEBUG_UTILS_NAME.as_ptr());
        }
        names
    }

    fn layers(&self) -> Vec<CString> {
        let mut names = Vec::new();
        if self.debug {
            names.push(CString::new("VK_LAYER_KHRONOS_validation").unwrap());
        }
        names
    }

    pub fn build(self) -> Result<Arc<Instance>, BackendError> {
        let entry = unsafe { ash::Entry::load()? };
        let extension_names = self.extensions();
        let layer_names = self.layers().iter().map(|x| x.as_ptr()).collect::<Vec<_>>();
        let info = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 3, 0));
        let desc = vk::InstanceCreateInfo::default()
            .application_info(&info)
            .enabled_layer_names(&layer_names)
            .enabled_extension_names(&extension_names);
        let instance = unsafe { entry.create_instance(&desc, None) }?;
        info!("Created a Vulkan instance");

        let debug = if self.debug {
            let utils = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                        | vk::DebugUtilsMessageTypeFlagsEXT::GENERAL,
                )
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .pfn_user_callback(Some(vulkan_debug_callback));
            let messanger = unsafe { utils.create_debug_utils_messenger(&info, None) }?;
            Some((utils, messanger))
        } else {
            None
        };

        Ok(Instance {
            entry,
            raw: instance,
            debug,
        }
        .into())
    }
}

impl Instance {
    pub fn debug_utils(&self) -> Option<&ash::ext::debug_utils::Instance> {
        if let Some((debug, _)) = &self.debug {
            Some(debug)
        } else {
            None
        }
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        if let Some((debug, messenger)) = self.debug.take() {
            unsafe { debug.destroy_debug_utils_messenger(messenger, None) };
        }
        unsafe {
            self.raw.destroy_instance(None);
        }
    }
}

unsafe extern "system" fn vulkan_debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    ty: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> u32 {
    let message = unsafe { (*data).message_as_c_str().unwrap().to_str().unwrap() };
    let ty = match ty {
        vk::DebugUtilsMessageTypeFlagsEXT::DEVICE_ADDRESS_BINDING => "Device address binding",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "Perf",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "Validation",
        _ => "",
    };

    match severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => error!("ERROR: {0}: {1}", ty, message),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => warn!("WARN {0}: {1}", ty, message),
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => trace!("{0}: {1}", ty, message),
        _ => info!("{0}: {1}", ty, message),
    };

    0
}
