use std::{ops::Deref, slice};

use ash::vk;

use crate::{
    BackendError,
    chunky_list::TempList,
    vulkan::{Device, SamplerDesc},
};

pub const MAX_DESCRIPTOR_SET_LAYOUTS: usize = 4;
pub const MAX_BINDLESS_RESOURCE_COUNT: usize = 64536;

#[derive(Debug)]
pub struct ShaderPipelineCommon {
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    pub descriptor_pool_sizes: Vec<gpu_descriptor::DescriptorTotalCount>,
    pub pipeline_bind_point: vk::PipelineBindPoint,
}

#[derive(Debug)]
pub struct ComputePipeline {
    pub pipeline: vk::Pipeline,
    pub common: ShaderPipelineCommon,
    pub group_size: [u32; 3],
}

impl Deref for ComputePipeline {
    type Target = ShaderPipelineCommon;

    fn deref(&self) -> &Self::Target {
        &self.common
    }
}

#[derive(Debug)]
pub struct RasterPipeline {
    pub pipeline: vk::Pipeline,
    pub common: ShaderPipelineCommon,
}

impl Deref for RasterPipeline {
    type Target = ShaderPipelineCommon;

    fn deref(&self) -> &Self::Target {
        &self.common
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorCount {
    Single,
    Count(u32),
    Bindless,
}

impl DescriptorCount {
    pub fn as_count(&self) -> u32 {
        match self {
            DescriptorCount::Single => 1,
            DescriptorCount::Count(count) => *count,
            DescriptorCount::Bindless => MAX_BINDLESS_RESOURCE_COUNT as u32,
        }
    }

    pub fn is_bindless(&self) -> bool {
        *self == Self::Bindless
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DescriptorDesc(pub u32, pub vk::DescriptorType, pub DescriptorCount);

#[derive(Debug, Clone, Copy)]
pub struct DescriptorSetDesc<'a>(pub &'a [(u32, &'a [DescriptorDesc])]);

impl ShaderPipelineCommon {
    fn new(
        device: &Device,
        desc: &DescriptorSetDesc,
        stage: vk::ShaderStageFlags,
        bind_point: vk::PipelineBindPoint,
    ) -> Result<ShaderPipelineCommon, BackendError> {
        let set_count = desc.0.iter().map(|x| x.0).max().unwrap_or(0);
        let mut set_layouts = vec![vk::DescriptorSetLayout::null(); set_count as usize];
        let mut pool_sizes =
            vec![gpu_descriptor::DescriptorTotalCount::default(); set_count as usize];
        let samplers = TempList::new();
        for (index, set) in desc.0.into_iter().copied() {
            let bindings = set
                .iter()
                .map(|desc| {
                    let mut binding = vk::DescriptorSetLayoutBinding::default()
                        .binding(desc.0)
                        .descriptor_type(desc.1)
                        .descriptor_count(desc.2.as_count())
                        .stage_flags(stage);
                    if desc.1 == vk::DescriptorType::SAMPLER
                        || desc.1 == vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                    {
                        let sampler = device
                            .get_sampler(SamplerDesc(
                                vk::Filter::LINEAR,
                                vk::SamplerMipmapMode::LINEAR,
                                vk::SamplerAddressMode::REPEAT,
                            ))
                            .unwrap();
                        binding =
                            binding.immutable_samplers(slice::from_ref(samplers.add(sampler)));
                    }
                    binding
                })
                .collect::<Vec<_>>();
            let mut count = gpu_descriptor::DescriptorTotalCount::default();
            let desc_count = set.iter().map(|x| x.0).max().unwrap_or(0);
            let mut flags = vec![vk::DescriptorBindingFlags::empty(); desc_count as usize];
            set.iter().for_each(|desc| {
                if desc.2.is_bindless() {
                    flags[desc.0 as usize] = vk::DescriptorBindingFlags::PARTIALLY_BOUND
                        | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
                }
                match desc.1 {
                    vk::DescriptorType::SAMPLED_IMAGE => count.sampled_image += desc.2.as_count(),
                    vk::DescriptorType::SAMPLER => count.sampler += desc.2.as_count(),
                    vk::DescriptorType::STORAGE_BUFFER => count.storage_buffer += desc.2.as_count(),
                    vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => {
                        count.storage_buffer_dynamic += desc.2.as_count()
                    }
                    vk::DescriptorType::STORAGE_IMAGE => count.storage_image += desc.2.as_count(),
                    vk::DescriptorType::UNIFORM_BUFFER => count.uniform_buffer += desc.2.as_count(),
                    vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC => {
                        count.uniform_buffer_dynamic += desc.2.as_count()
                    }
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER => {
                        count.combined_image_sampler += desc.2.as_count()
                    }
                    _ => panic!("Descriptor type {:?} not supported", desc.1),
                }
            });
            pool_sizes[index as usize] = count;
            let flag = if set.iter().any(|x| x.2.is_bindless()) {
                vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL
            } else {
                vk::DescriptorSetLayoutCreateFlags::empty()
            };
            let mut binding_flags =
                vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
            let info = vk::DescriptorSetLayoutCreateInfo::default()
                .bindings(&bindings)
                .flags(flag)
                .push_next(&mut binding_flags);
            let set_layout = unsafe { device.raw.create_descriptor_set_layout(&info, None) }?;
            set_layouts[index as usize] = set_layout;
        }

        let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        let pipeline_layout = unsafe { device.raw.create_pipeline_layout(&info, None) }?;
        Ok(Self {
            pipeline_layout,
            descriptor_set_layouts: set_layouts,
            descriptor_pool_sizes: pool_sizes,
            pipeline_bind_point: bind_point,
        })
    }

    fn free(mut self, device: &ash::Device) {
        unsafe { device.destroy_pipeline_layout(self.pipeline_layout, None) };
        self.descriptor_set_layouts
            .drain(..)
            .for_each(|layout| unsafe {
                device.destroy_descriptor_set_layout(layout, None);
            });
    }
}
