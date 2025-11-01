use thiserror::Error;

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("Failed to load Vulkan")]
    LoadingError,
    #[error("Vulkan error: {0:?}")]
    VulkanError(#[from] ash::vk::Result),
    #[error("Can't get display/window handle: {0:?}")]
    RawWindowHandleError(raw_window_handle::HandleError)
}

impl From<ash::LoadingError> for BackendError {
    fn from(_: ash::LoadingError) -> Self {
        Self::LoadingError
    }
}

impl From<raw_window_handle::HandleError> for BackendError {
    fn from(value: raw_window_handle::HandleError) -> Self {
        Self::RawWindowHandleError(value)
    }
}
