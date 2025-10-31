use thiserror::Error;

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("Failed to load Vulkan")]
    LoadingError,
    #[error("Vulkan error: {0:?}")]
    VulkanError(ash::vk::Result)
}

impl From<ash::vk::Result> for BackendError {
    fn from(value: ash::vk::Result) -> Self {
        return Self::VulkanError(value)
    }
}

impl From<ash::LoadingError> for BackendError {
    fn from(_: ash::LoadingError) -> Self {
        return Self::LoadingError;
    }
}