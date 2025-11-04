use std::{ffi::CString, fmt::Debug, ops::Deref, slice, sync::Arc};

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
    device: Arc<Device>,
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
    device: Arc<Device>,
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

    fn free(&mut self, device: &ash::Device) {
        unsafe { device.destroy_pipeline_layout(self.pipeline_layout, None) };
        self.descriptor_set_layouts
            .drain(..)
            .for_each(|layout| unsafe {
                device.destroy_descriptor_set_layout(layout, None);
            });
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ClearAttachment {
    None,
    Color([u8; 4]),
    Depth(f32),
}

#[derive(Debug, Clone, Copy)]
pub struct AttachmentDesc {
    pub format: vk::Format,
    pub load: vk::AttachmentLoadOp,
    pub store: vk::AttachmentStoreOp,
    pub clean: ClearAttachment,
}

impl Default for AttachmentDesc {
    fn default() -> Self {
        Self {
            format: vk::Format::UNDEFINED,
            load: vk::AttachmentLoadOp::LOAD,
            store: vk::AttachmentStoreOp::STORE,
            clean: ClearAttachment::None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RenderTargetDesc<'a> {
    pub color: &'a [AttachmentDesc],
    pub depth: Option<AttachmentDesc>,
}

#[derive(Debug, Clone, Copy)]
pub struct RasterPipelineDesc<'a> {
    pub layout: &'a DescriptorSetDesc<'a>,
    pub render_target: &'a RenderTargetDesc<'a>,
    pub primitive: vk::PrimitiveTopology,
    pub depth_test: Option<vk::CompareOp>,
    pub depth_write: bool,
    pub cull: vk::CullModeFlags,
    pub front: vk::FrontFace,
    pub blend: Option<(vk::BlendOp, vk::BlendFactor, vk::BlendFactor)>,
}

impl<'a> RasterPipelineDesc<'a> {
    pub fn new(layout: &'a DescriptorSetDesc<'a>, render_target: &'a RenderTargetDesc<'a>) -> Self {
        Self {
            layout,
            render_target,
            primitive: vk::PrimitiveTopology::TRIANGLE_LIST,
            depth_test: None,
            depth_write: true,
            front: vk::FrontFace::CLOCKWISE,
            cull: vk::CullModeFlags::BACK,
            blend: None,
        }
    }

    pub fn primitive(mut self, value: vk::PrimitiveTopology) -> Self {
        self.primitive = value;
        self
    }

    pub fn depth_test(mut self, value: vk::CompareOp) -> Self {
        self.depth_test = Some(value);
        self
    }

    pub fn depth_write(mut self, value: bool) -> Self {
        self.depth_write = value;
        self
    }

    pub fn front(mut self, value: vk::FrontFace) -> Self {
        self.front = value;
        self
    }

    pub fn cull(mut self, value: vk::CullModeFlags) -> Self {
        self.cull = value;
        self
    }

    pub fn blend(mut self, op: vk::BlendOp, src: vk::BlendFactor, dst: vk::BlendFactor) -> Self {
        self.blend = Some((op, src, dst));
        self
    }
}

#[derive(Debug, Clone)]
pub struct Shader {
    pub module: vk::ShaderModule,
    pub stage: vk::ShaderStageFlags,
    pub entry: CString,
}

impl RasterPipeline {
    pub fn new(
        device: Arc<Device>,
        shaders: &[Shader],
        desc: RasterPipelineDesc,
    ) -> Result<Self, BackendError> {
        let common = ShaderPipelineCommon::new(
            &device,
            desc.layout,
            vk::ShaderStageFlags::ALL_GRAPHICS,
            vk::PipelineBindPoint::GRAPHICS,
        )?;
        let shaders = shaders
            .iter()
            .map(|shader| {
                vk::PipelineShaderStageCreateInfo::default()
                    .module(shader.module)
                    .stage(shader.stage)
                    .name(&shader.entry)
            })
            .collect::<Vec<_>>();
        let assembly_state =
            vk::PipelineInputAssemblyStateCreateInfo::default().topology(desc.primitive);
        let raster_state = vk::PipelineRasterizationStateCreateInfo::default()
            .cull_mode(desc.cull)
            .front_face(desc.front);
        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(desc.depth_test.is_some())
            .depth_write_enable(desc.depth_write)
            .depth_compare_op(desc.depth_test.unwrap_or(vk::CompareOp::ALWAYS));
        let blend_attachments = if let Some((op, src, dst)) = desc.blend {
            // fixme: alpha?
            let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(true)
                .color_blend_op(op)
                .src_color_blend_factor(src)
                .src_alpha_blend_factor(src)
                .dst_color_blend_factor(dst)
                .dst_alpha_blend_factor(dst);
            vec![blend_attachment; desc.render_target.color.len()]
        } else {
            vec![vk::PipelineColorBlendAttachmentState::default(); desc.render_target.color.len()]
        };
        let blend_state =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);
        let color_formats = desc
            .render_target
            .color
            .iter()
            .map(|x| x.format)
            .collect::<Vec<_>>();
        let mut rendering =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);
        if let Some(depth) = desc.render_target.depth {
            rendering = rendering.depth_attachment_format(depth.format);
        }
        let info = vk::GraphicsPipelineCreateInfo::default()
            .layout(common.pipeline_layout)
            .depth_stencil_state(&depth_stencil_state)
            .rasterization_state(&raster_state)
            .stages(&shaders)
            .input_assembly_state(&assembly_state)
            .color_blend_state(&blend_state)
            .dynamic_state(&dynamic_state)
            .push_next(&mut rendering);
        let pipeline = unsafe {
            device.raw.create_graphics_pipelines(
                vk::PipelineCache::null(),
                slice::from_ref(&info),
                None,
            )
        }
        .map_err(|(_, err)| BackendError::VulkanError(err))?[0];
        Ok(Self {
            device,
            pipeline,
            common,
        })
    }
}

impl Drop for RasterPipeline {
    fn drop(&mut self) {
        unsafe { self.device.raw.destroy_pipeline(self.pipeline, None) };
        self.common.free(&self.device.raw);
    }
}
