use std::{collections::HashMap, sync::Arc};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::vulkan::{
    ComputePipeline, ComputePipelineDesc, Device, RasterPipeline, RasterPipelineDesc,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputePipelineHandle(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RasterPipelineHandle(usize);

#[derive(Debug)]
pub struct PipelineCache {
    device: Arc<Device>,
    raster_pipeline_desc: HashMap<RasterPipelineHandle, RasterPipelineDesc>,
    compute_pipeline_desc: HashMap<ComputePipelineHandle, ComputePipelineDesc>,
    raster_pipelines: Vec<Option<Arc<RasterPipeline>>>,
    compute_pipelines: Vec<Option<Arc<ComputePipeline>>>,
}

impl PipelineCache {
    pub fn new(device: Arc<Device>) -> Self {
        Self {
            device,
            raster_pipeline_desc: Default::default(),
            compute_pipeline_desc: Default::default(),
            raster_pipelines: Default::default(),
            compute_pipelines: Default::default(),
        }
    }

    pub fn register_raster_pipeline(&mut self, desc: RasterPipelineDesc) -> RasterPipelineHandle {
        let handle = RasterPipelineHandle(self.raster_pipelines.len());
        self.raster_pipelines.push(None);
        self.raster_pipeline_desc.insert(handle, desc);
        handle
    }

    pub fn register_compute_pipeline(
        &mut self,
        desc: ComputePipelineDesc,
    ) -> ComputePipelineHandle {
        let handle = ComputePipelineHandle(self.raster_pipelines.len());
        self.compute_pipelines.push(None);
        self.compute_pipeline_desc.insert(handle, desc);
        handle
    }

    pub fn update_raster_pipeline(
        &mut self,
        handle: RasterPipelineHandle,
        desc: RasterPipelineDesc,
    ) {
        self.raster_pipeline_desc.insert(handle, desc);
        self.raster_pipelines[handle.0] = None;
    }

    pub fn update_compute_pipeline(
        &mut self,
        handle: ComputePipelineHandle,
        desc: ComputePipelineDesc,
    ) {
        self.compute_pipeline_desc.insert(handle, desc);
        self.compute_pipelines[handle.0] = None;
    }

    pub fn compile_all_pieplines(&mut self) {
        let raster_to_compile = self
            .raster_pipelines
            .iter()
            .enumerate()
            .filter_map(|(index, pipeline)| {
                if pipeline.is_none() {
                    Some(RasterPipelineHandle(index))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let compute_to_compile = self
            .compute_pipelines
            .iter()
            .enumerate()
            .filter_map(|(index, pipeline)| {
                if pipeline.is_none() {
                    Some(ComputePipelineHandle(index))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let raster_compiled = raster_to_compile
            .par_iter()
            .map(|handle| {
                let desc = self.raster_pipeline_desc.get(handle).unwrap();
                let pipeline = RasterPipeline::new(self.device.clone(), desc).unwrap();
                (handle, pipeline)
            })
            .collect::<Vec<_>>();
        let compute_compiled = compute_to_compile
            .par_iter()
            .map(|handle| {
                let desc = self.compute_pipeline_desc.get(handle).unwrap();
                let pipeline = ComputePipeline::new(self.device.clone(), desc).unwrap();
                (handle, pipeline)
            })
            .collect::<Vec<_>>();
        for (handle, pipeline) in raster_compiled {
            self.raster_pipelines[handle.0] = Some(Arc::new(pipeline));
        }
        for (handle, pipeline) in compute_compiled {
            self.compute_pipelines[handle.0] = Some(Arc::new(pipeline));
        }
    }

    pub fn get_raster_pipeline(&self, handle: RasterPipelineHandle) -> Arc<RasterPipeline> {
        self.raster_pipelines
            .get(handle.0)
            .cloned()
            .expect("Wrong pipeline handle")
            .expect("Pipelines must be compiled first")
    }

    pub fn get_compute_pipeline(&self, handle: ComputePipelineHandle) -> Arc<ComputePipeline> {
        self.compute_pipelines
            .get(handle.0)
            .cloned()
            .expect("Wrong pipeline handle")
            .expect("Pipelines must be compiled first")
    }
}
