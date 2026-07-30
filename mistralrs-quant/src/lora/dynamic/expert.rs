use std::sync::{Arc, OnceLock};

#[cfg(feature = "cuda")]
use std::sync::{Mutex, MutexGuard};

use candle_core::{DType, Device, Result, Tensor};

use crate::Shard;

use super::{
    current_lora_execution, LoraExecution, LoraRuntimeId, LoraSiteHandle, LoraSiteKey, LoraWeights,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoraExpertProjection {
    Gate,
    Up,
    Down,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LoraGateUpOrder {
    #[default]
    Concatenated,
    Interleaved,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoraExpertProjectionNames {
    gate: Arc<str>,
    up: Arc<str>,
    down: Arc<str>,
}

impl LoraExpertProjectionNames {
    pub fn new(
        gate: impl Into<Arc<str>>,
        up: impl Into<Arc<str>>,
        down: impl Into<Arc<str>>,
    ) -> Self {
        Self {
            gate: gate.into(),
            up: up.into(),
            down: down.into(),
        }
    }

    pub fn name(&self, projection: LoraExpertProjection) -> &str {
        match projection {
            LoraExpertProjection::Gate => &self.gate,
            LoraExpertProjection::Up => &self.up,
            LoraExpertProjection::Down => &self.down,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoraExpertSiteSpec {
    num_experts: usize,
    hidden_size: usize,
    intermediate_size: usize,
    projection_names: LoraExpertProjectionNames,
    gate_up_output_shard: Shard,
    down_input_shard: Shard,
    gate_up_order: LoraGateUpOrder,
}

impl LoraExpertSiteSpec {
    pub fn new(
        num_experts: usize,
        hidden_size: usize,
        intermediate_size: usize,
        projection_names: LoraExpertProjectionNames,
        gate_up_output_shard: Shard,
        down_input_shard: Shard,
    ) -> Result<Self> {
        if num_experts == 0 || hidden_size == 0 || intermediate_size == 0 {
            candle_core::bail!("expert LoRA dimensions must be nonzero");
        }
        let gate_up_size = sharded_size(intermediate_size, gate_up_output_shard)?;
        let down_size = sharded_size(intermediate_size, down_input_shard)?;
        if gate_up_size != down_size {
            candle_core::bail!(
                "expert LoRA gate/up output size {gate_up_size} does not match down input size {down_size}"
            );
        }
        Ok(Self {
            num_experts,
            hidden_size,
            intermediate_size,
            projection_names,
            gate_up_output_shard,
            down_input_shard,
            gate_up_order: LoraGateUpOrder::Concatenated,
        })
    }

    pub fn with_gate_up_order(mut self, order: LoraGateUpOrder) -> Self {
        self.gate_up_order = order;
        self
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    pub fn local_intermediate_size(&self) -> usize {
        sharded_size(self.intermediate_size, self.gate_up_output_shard)
            .expect("expert LoRA shard was validated at construction")
    }

    pub fn name(&self, projection: LoraExpertProjection) -> &str {
        self.projection_names.name(projection)
    }

    pub fn projection_names(&self) -> &LoraExpertProjectionNames {
        &self.projection_names
    }

    pub fn gate_up_output_shard(&self) -> Shard {
        self.gate_up_output_shard
    }

    pub fn down_input_shard(&self) -> Shard {
        self.down_input_shard
    }

    pub fn gate_up_order(&self) -> LoraGateUpOrder {
        self.gate_up_order
    }

    pub(super) fn projection_shape(&self, projection: LoraExpertProjection) -> (usize, usize) {
        match projection {
            LoraExpertProjection::Gate | LoraExpertProjection::Up => {
                (self.hidden_size, self.local_intermediate_size())
            }
            LoraExpertProjection::Down => (self.local_intermediate_size(), self.hidden_size),
        }
    }
}

fn sharded_size(size: usize, shard: Shard) -> Result<usize> {
    match shard {
        Shard::Simple {
            rank, world_size, ..
        } => {
            if world_size == 0 || rank >= world_size || !size.is_multiple_of(world_size) {
                candle_core::bail!("invalid expert LoRA shard");
            }
            Ok(size / world_size)
        }
        Shard::Offset { offset, len, .. } => {
            if len == 0 || offset.checked_add(len).is_none_or(|end| end > size) {
                candle_core::bail!("invalid expert LoRA shard");
            }
            Ok(len)
        }
    }
}

#[derive(Debug)]
pub struct LoraExpertSiteHandle {
    runtime_id: LoraRuntimeId,
    key: LoraSiteKey,
    spec: LoraExpertSiteSpec,
    activation_dtype: DType,
    device: Device,
    id: OnceLock<u32>,
}

impl LoraExpertSiteHandle {
    pub(super) fn new(
        runtime_id: LoraRuntimeId,
        key: LoraSiteKey,
        spec: LoraExpertSiteSpec,
        activation_dtype: DType,
        device: Device,
    ) -> Self {
        Self {
            runtime_id,
            key,
            spec,
            activation_dtype,
            device,
            id: OnceLock::new(),
        }
    }

    pub fn runtime_id(&self) -> LoraRuntimeId {
        self.runtime_id
    }

    pub fn key(&self) -> &LoraSiteKey {
        &self.key
    }

    pub fn spec(&self) -> &LoraExpertSiteSpec {
        &self.spec
    }

    pub fn activation_dtype(&self) -> DType {
        self.activation_dtype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn id(&self) -> Result<u32> {
        self.id
            .get()
            .copied()
            .ok_or_else(|| candle_core::Error::msg("LoRA layer registry has not been finalized"))
    }

    pub(super) fn assign_id(&self, id: u32) -> Result<()> {
        self.id
            .set(id)
            .map_err(|_| candle_core::Error::msg("LoRA expert site ID was already assigned"))
    }
}

#[derive(Clone, Debug)]
pub struct LoraExpertProjectionWeights {
    a: Tensor,
    b: Tensor,
    scales: Tensor,
}

impl LoraExpertProjectionWeights {
    pub fn new(a: Tensor, b: Tensor, scales: Tensor) -> Result<Self> {
        Self::new_inner(a, b, scales, true)
    }

    pub(crate) fn new_loaded(a: Tensor, b: Tensor, scales: Tensor) -> Result<Self> {
        Self::new_inner(a, b, scales, false)
    }

    fn new_inner(a: Tensor, b: Tensor, scales: Tensor, check_finite: bool) -> Result<Self> {
        let (experts, rank, _) = a.dims3()?;
        let (b_experts, _, b_rank) = b.dims3()?;
        if experts == 0 || rank == 0 {
            candle_core::bail!("expert LoRA expert count and rank must be nonzero");
        }
        if b_experts != experts || b_rank != rank {
            candle_core::bail!(
                "expert LoRA A shape {:?} is incompatible with B shape {:?}",
                a.dims(),
                b.dims()
            );
        }
        if scales.dims1()? != experts || scales.dtype() != DType::F32 {
            candle_core::bail!("expert LoRA scales must be F32 with shape [{experts}]");
        }
        if a.dtype() != b.dtype() {
            candle_core::bail!(
                "expert LoRA A dtype {:?} does not match B dtype {:?}",
                a.dtype(),
                b.dtype()
            );
        }
        let location = a.device().location();
        if b.device().location() != location || scales.device().location() != location {
            candle_core::bail!("expert LoRA A, B, and scales must be on the same device");
        }
        if check_finite
            && scales
                .to_device(&Device::Cpu)?
                .to_vec1::<f32>()?
                .iter()
                .any(|scale| !scale.is_finite())
        {
            candle_core::bail!("expert LoRA scales must be finite");
        }
        Ok(Self {
            a: a.contiguous()?,
            b: b.contiguous()?,
            scales: scales.contiguous()?,
        })
    }

    pub fn a(&self) -> &Tensor {
        &self.a
    }

    pub fn b(&self) -> &Tensor {
        &self.b
    }

    pub fn scales(&self) -> &Tensor {
        &self.scales
    }

    pub fn rank(&self) -> usize {
        self.a.dim(1).expect("expert LoRA A was validated")
    }
}

#[derive(Clone, Debug)]
pub struct LoraExpertWeights {
    runtime_id: LoraRuntimeId,
    site_key: LoraSiteKey,
    gate: Option<LoraExpertProjectionWeights>,
    up: Option<LoraExpertProjectionWeights>,
    down: Option<LoraExpertProjectionWeights>,
    #[cfg(feature = "cuda")]
    cuda_table: Arc<ExpertCudaTableCache>,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
struct ExpertCudaTableCache {
    table: Mutex<Option<Arc<super::RoutedLoraCudaWeightTable>>>,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for ExpertCudaTableCache {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("ExpertCudaTableCache")
    }
}

impl LoraExpertWeights {
    pub fn new(
        site: &LoraExpertSiteHandle,
        gate: Option<LoraExpertProjectionWeights>,
        up: Option<LoraExpertProjectionWeights>,
        down: Option<LoraExpertProjectionWeights>,
    ) -> Result<Self> {
        if gate.is_none() && up.is_none() && down.is_none() {
            candle_core::bail!("expert LoRA weights must contain at least one projection");
        }
        for (projection, weights) in [
            (LoraExpertProjection::Gate, gate.as_ref()),
            (LoraExpertProjection::Up, up.as_ref()),
            (LoraExpertProjection::Down, down.as_ref()),
        ] {
            let Some(weights) = weights else {
                continue;
            };
            validate_projection_weights(site, projection, weights)?;
        }
        Ok(Self {
            runtime_id: site.runtime_id(),
            site_key: site.key().clone(),
            gate,
            up,
            down,
            #[cfg(feature = "cuda")]
            cuda_table: Arc::new(ExpertCudaTableCache::default()),
        })
    }

    pub(crate) fn validate_for_site(&self, site: &LoraExpertSiteHandle) -> Result<()> {
        if self.runtime_id != site.runtime_id() || &self.site_key != site.key() {
            candle_core::bail!("expert LoRA weights belong to a different site");
        }
        for (projection, weights) in [
            (LoraExpertProjection::Gate, self.gate.as_ref()),
            (LoraExpertProjection::Up, self.up.as_ref()),
            (LoraExpertProjection::Down, self.down.as_ref()),
        ] {
            if let Some(weights) = weights {
                validate_projection_weights(site, projection, weights)?;
            }
        }
        Ok(())
    }

    pub fn projection(
        &self,
        projection: LoraExpertProjection,
    ) -> Option<&LoraExpertProjectionWeights> {
        match projection {
            LoraExpertProjection::Gate => self.gate.as_ref(),
            LoraExpertProjection::Up => self.up.as_ref(),
            LoraExpertProjection::Down => self.down.as_ref(),
        }
    }

    pub fn gate(&self) -> Option<&LoraExpertProjectionWeights> {
        self.gate.as_ref()
    }

    pub fn up(&self) -> Option<&LoraExpertProjectionWeights> {
        self.up.as_ref()
    }

    pub fn down(&self) -> Option<&LoraExpertProjectionWeights> {
        self.down.as_ref()
    }

    #[cfg(feature = "cuda")]
    pub(super) fn cuda_table(
        &self,
    ) -> MutexGuard<'_, Option<Arc<super::RoutedLoraCudaWeightTable>>> {
        self.cuda_table
            .table
            .lock()
            .expect("expert LoRA CUDA table cache poisoned")
    }
}

fn validate_projection_weights(
    site: &LoraExpertSiteHandle,
    projection: LoraExpertProjection,
    weights: &LoraExpertProjectionWeights,
) -> Result<()> {
    let spec = site.spec();
    let (input_features, output_features) = spec.projection_shape(projection);
    let expected_a = [spec.num_experts(), weights.rank(), input_features];
    let expected_b = [spec.num_experts(), output_features, weights.rank()];
    if weights.a().dims() != expected_a || weights.b().dims() != expected_b {
        candle_core::bail!(
            "expert LoRA {:?} weights have shapes A={:?}, B={:?}, expected A={expected_a:?}, B={expected_b:?}",
            projection,
            weights.a().dims(),
            weights.b().dims()
        );
    }
    if weights.a().dtype() != site.activation_dtype() {
        candle_core::bail!(
            "expert LoRA {:?} weights must use activation dtype {:?}, got {:?}",
            projection,
            site.activation_dtype(),
            weights.a().dtype()
        );
    }
    if weights.a().device().location() != site.device().location() {
        candle_core::bail!("expert LoRA weights and site must be on the same device");
    }
    Ok(())
}

#[derive(Debug, Default)]
pub struct DynamicLoraWeights {
    pub linear: Vec<(Arc<LoraSiteHandle>, LoraWeights)>,
    pub experts: Vec<(Arc<LoraExpertSiteHandle>, LoraExpertWeights)>,
}

impl From<Vec<(Arc<LoraSiteHandle>, LoraWeights)>> for DynamicLoraWeights {
    fn from(linear: Vec<(Arc<LoraSiteHandle>, LoraWeights)>) -> Self {
        Self {
            linear,
            experts: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoraExpertInputMode {
    TokenRows,
    RoutedRows,
}

#[derive(Debug)]
pub struct LoraExpertDelta<'a> {
    pub(super) projection: LoraExpertProjection,
    pub(super) input: &'a Tensor,
    pub(super) base_output: Tensor,
    pub(super) topk_ids: &'a Tensor,
    pub(super) routed_weights: Option<&'a Tensor>,
    pub(super) input_mode: LoraExpertInputMode,
}

impl<'a> LoraExpertDelta<'a> {
    pub fn new(
        projection: LoraExpertProjection,
        input: &'a Tensor,
        base_output: Tensor,
        topk_ids: &'a Tensor,
        input_mode: LoraExpertInputMode,
    ) -> Self {
        Self {
            projection,
            input,
            base_output,
            topk_ids,
            routed_weights: None,
            input_mode,
        }
    }

    pub fn with_routed_weights(mut self, routed_weights: &'a Tensor) -> Self {
        self.routed_weights = Some(routed_weights);
        self
    }
}

#[derive(Debug)]
pub struct LoraExpertExecution {
    execution: Arc<LoraExecution>,
    site: Arc<LoraExpertSiteHandle>,
}

impl LoraExpertExecution {
    pub fn current(site: &Arc<LoraExpertSiteHandle>) -> Result<Option<Self>> {
        let Some(execution) = current_lora_execution(site.runtime_id()) else {
            return Ok(None);
        };
        if !execution.expert_site_is_active(site)? {
            return Ok(None);
        }
        Ok(Some(Self {
            execution,
            site: site.clone(),
        }))
    }

    pub fn site(&self) -> &Arc<LoraExpertSiteHandle> {
        &self.site
    }

    pub fn add_delta(
        &self,
        projection: LoraExpertProjection,
        input: &Tensor,
        base_output: Tensor,
        topk_ids: &Tensor,
        routed_weights: Option<&Tensor>,
        input_mode: LoraExpertInputMode,
    ) -> Result<Tensor> {
        let mut delta = LoraExpertDelta::new(projection, input, base_output, topk_ids, input_mode);
        if let Some(routed_weights) = routed_weights {
            delta = delta.with_routed_weights(routed_weights);
        }
        self.add_delta_inner(delta, false)
    }

    #[doc(hidden)]
    pub fn add_delta_owned(
        &self,
        projection: LoraExpertProjection,
        input: &Tensor,
        base_output: Tensor,
        topk_ids: &Tensor,
        routed_weights: Option<&Tensor>,
        input_mode: LoraExpertInputMode,
    ) -> Result<Tensor> {
        let mut delta = LoraExpertDelta::new(projection, input, base_output, topk_ids, input_mode);
        if let Some(routed_weights) = routed_weights {
            delta = delta.with_routed_weights(routed_weights);
        }
        self.add_delta_inner(delta, true)
    }

    fn add_delta_inner(&self, delta: LoraExpertDelta<'_>, in_place: bool) -> Result<Tensor> {
        #[cfg(not(feature = "cuda"))]
        let _ = in_place;
        #[cfg(feature = "cuda")]
        if let Some(output) =
            super::expert_cuda::try_add_delta(&self.execution, &self.site, &delta, in_place)?
        {
            return Ok(output);
        }
        add_expert_delta_reference(&self.execution, &self.site, delta)
    }

    pub fn add_gate_up_delta(
        &self,
        input: &Tensor,
        base_gate_up: Tensor,
        topk_ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        self.add_gate_up_delta_inner(input, base_gate_up, topk_ids, false)
    }

    #[doc(hidden)]
    pub fn add_gate_up_delta_owned(
        &self,
        input: &Tensor,
        base_gate_up: Tensor,
        topk_ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        self.add_gate_up_delta_inner(input, base_gate_up, topk_ids, true)
    }

    #[doc(hidden)]
    pub fn add_gate_up_delta_combined_owned(
        &self,
        input: &Tensor,
        base_gate_up: Tensor,
        topk_ids: &Tensor,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Some(output) = super::expert_cuda::try_add_gate_up_delta_combined(
            &self.execution,
            &self.site,
            input,
            &base_gate_up,
            topk_ids,
            true,
        )? {
            return Ok(output);
        }

        let (gate, up) = split_gate_up(&self.site, base_gate_up, topk_ids)?;
        let gate = self.add_delta(
            LoraExpertProjection::Gate,
            input,
            gate,
            topk_ids,
            None,
            LoraExpertInputMode::TokenRows,
        )?;
        let up = self.add_delta(
            LoraExpertProjection::Up,
            input,
            up,
            topk_ids,
            None,
            LoraExpertInputMode::TokenRows,
        )?;
        Tensor::cat(&[&gate, &up], candle_core::D::Minus1)
    }

    fn add_gate_up_delta_inner(
        &self,
        input: &Tensor,
        base_gate_up: Tensor,
        topk_ids: &Tensor,
        in_place: bool,
    ) -> Result<(Tensor, Tensor)> {
        #[cfg(not(feature = "cuda"))]
        let _ = in_place;
        #[cfg(feature = "cuda")]
        if let Some((gate, up)) = super::expert_cuda::try_add_gate_up_delta(
            &self.execution,
            &self.site,
            input,
            &base_gate_up,
            topk_ids,
            in_place,
        )? {
            return Ok((gate, up));
        }

        let (gate, up) = split_gate_up(&self.site, base_gate_up, topk_ids)?;
        let gate = self.add_delta(
            LoraExpertProjection::Gate,
            input,
            gate,
            topk_ids,
            None,
            LoraExpertInputMode::TokenRows,
        )?;
        let up = self.add_delta(
            LoraExpertProjection::Up,
            input,
            up,
            topk_ids,
            None,
            LoraExpertInputMode::TokenRows,
        )?;
        Ok((gate, up))
    }
}

fn split_gate_up(
    site: &LoraExpertSiteHandle,
    gate_up: Tensor,
    topk_ids: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (num_tokens, top_k) = topk_ids.dims2()?;
    let intermediate = site.spec().local_intermediate_size();
    if gate_up.elem_count() != num_tokens * top_k * intermediate * 2 {
        candle_core::bail!(
            "expert LoRA gate/up output must contain [tokens, top_k, 2 * intermediate] elements"
        );
    }
    let gate_up = gate_up.reshape((num_tokens, top_k, intermediate * 2))?;
    match site.spec().gate_up_order() {
        LoraGateUpOrder::Concatenated => Ok((
            gate_up
                .narrow(candle_core::D::Minus1, 0, intermediate)?
                .contiguous()?,
            gate_up
                .narrow(candle_core::D::Minus1, intermediate, intermediate)?
                .contiguous()?,
        )),
        LoraGateUpOrder::Interleaved => {
            let gate_up = gate_up.reshape((num_tokens, top_k, intermediate, 2))?;
            Ok((
                gate_up
                    .narrow(candle_core::D::Minus1, 0, 1)?
                    .squeeze(candle_core::D::Minus1)?
                    .contiguous()?,
                gate_up
                    .narrow(candle_core::D::Minus1, 1, 1)?
                    .squeeze(candle_core::D::Minus1)?
                    .contiguous()?,
            ))
        }
    }
}

pub fn add_expert_delta_reference(
    execution: &LoraExecution,
    site: &LoraExpertSiteHandle,
    delta: LoraExpertDelta<'_>,
) -> Result<Tensor> {
    let LoraExpertDelta {
        projection,
        input,
        base_output,
        topk_ids,
        routed_weights,
        input_mode,
    } = delta;
    if execution.runtime_id() != site.runtime_id() {
        candle_core::bail!("expert LoRA site and execution belong to different runtimes");
    }
    let (num_tokens, top_k) = topk_ids.dims2()?;
    let output_features = base_output.dim(candle_core::D::Minus1)?;
    if base_output.dims() != [num_tokens, top_k, output_features] {
        candle_core::bail!(
            "expert LoRA base output must have shape [tokens, top_k, output_features]"
        );
    }
    if execution.row_slots().len() != num_tokens {
        candle_core::bail!(
            "expert LoRA route count {} does not match token count {num_tokens}",
            execution.row_slots().len()
        );
    }
    if let Some(routed_weights) = routed_weights {
        if routed_weights.dims2()? != (num_tokens, top_k) {
            candle_core::bail!("expert LoRA routed weights must have shape [tokens, top_k]");
        }
        if routed_weights.device().location() != input.device().location() {
            candle_core::bail!("expert LoRA routed weights must share the input device");
        }
    }
    let input_features = match input_mode {
        LoraExpertInputMode::TokenRows => {
            let (tokens, features) = input.dims2()?;
            if tokens != num_tokens {
                candle_core::bail!("expert LoRA token input count does not match routing");
            }
            features
        }
        LoraExpertInputMode::RoutedRows => {
            let (tokens, routes, features) = input.dims3()?;
            if tokens != num_tokens || routes != top_k {
                candle_core::bail!("expert LoRA routed input shape does not match routing");
            }
            features
        }
    };
    let (expected_input, expected_output) = site.spec().projection_shape(projection);
    if input_features != expected_input || output_features != expected_output {
        candle_core::bail!(
            "expert LoRA {:?} dimensions input={input_features}, output={output_features}, expected input={expected_input}, output={expected_output}",
            projection
        );
    }
    if input.dtype() != base_output.dtype()
        || input.device().location() != base_output.device().location()
        || input.device().location() != topk_ids.device().location()
    {
        candle_core::bail!(
            "expert LoRA inputs, routes, and output must share a device and activation dtype"
        );
    }
    if input.dtype() != site.activation_dtype()
        || input.device().location() != site.device().location()
    {
        candle_core::bail!("expert LoRA inputs do not match the registered site");
    }

    let topk_ids = if topk_ids.dtype() == DType::U32 {
        topk_ids.clone()
    } else {
        topk_ids.to_dtype(DType::U32)?
    };
    let mut output = base_output
        .reshape((num_tokens * top_k, output_features))?
        .contiguous()?;
    for slot in execution.rows_by_slot().keys() {
        let Some(adapter) = execution.expert_weights(site, *slot)? else {
            continue;
        };
        let Some(weights) = adapter.projection(projection) else {
            continue;
        };
        let token_indices = execution
            .row_indices(*slot, input.device())?
            .expect("active expert LoRA slot has routed rows");
        let active_tokens = token_indices.elem_count();
        let expert_ids = topk_ids
            .index_select(&token_indices, 0)?
            .reshape(active_tokens * top_k)?;
        let a = weights.a().index_select(&expert_ids, 0)?;
        let b = weights.b().index_select(&expert_ids, 0)?;
        let selected_input = match input_mode {
            LoraExpertInputMode::TokenRows => input
                .index_select(&token_indices, 0)?
                .unsqueeze(1)?
                .broadcast_as((active_tokens, top_k, input_features))?,
            LoraExpertInputMode::RoutedRows => input.index_select(&token_indices, 0)?,
        }
        .reshape((active_tokens * top_k, 1, input_features))?;
        let hidden = selected_input.matmul(&a.transpose(1, 2)?)?;
        let mut delta = hidden.matmul(&b.transpose(1, 2)?)?.squeeze(1)?;
        let scales = weights
            .scales()
            .index_select(&expert_ids, 0)?
            .to_dtype(delta.dtype())?
            .unsqueeze(1)?;
        delta = delta.broadcast_mul(&scales)?;
        if let Some(routed_weights) = routed_weights {
            let selected_weights = routed_weights
                .index_select(&token_indices, 0)?
                .reshape(active_tokens * top_k)?
                .to_dtype(delta.dtype())?
                .unsqueeze(1)?;
            delta = delta.broadcast_mul(&selected_weights)?;
        }
        let route_indices = routed_row_indices(
            execution
                .rows_by_slot()
                .get(slot)
                .expect("active expert LoRA slot has host rows"),
            top_k,
            input.device(),
        )?;
        output = output.index_add(&route_indices, &delta, 0)?;
    }
    output.reshape((num_tokens, top_k, output_features))
}

fn routed_row_indices(rows: &[usize], top_k: usize, device: &Device) -> Result<Tensor> {
    let mut indices = Vec::with_capacity(rows.len() * top_k);
    for row in rows {
        let start = row
            .checked_mul(top_k)
            .ok_or_else(|| candle_core::Error::msg("expert LoRA route index overflow"))?;
        for route in 0..top_k {
            indices.push(u32::try_from(start + route).map_err(candle_core::Error::wrap)?);
        }
    }
    Tensor::from_vec(indices, rows.len() * top_k, device)
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::*;
    use crate::{LoraLayerRegistry, LoraSiteKey};

    fn site() -> Result<(LoraLayerRegistry, Arc<LoraExpertSiteHandle>)> {
        let registry = LoraLayerRegistry::new();
        let site = registry.register_expert(
            LoraSiteKey::new("model.layers.0.mlp.experts"),
            LoraExpertSiteSpec::new(
                2,
                2,
                2,
                LoraExpertProjectionNames::new("gate_proj", "up_proj", "down_proj"),
                Shard::default(),
                Shard::default(),
            )?,
            DType::F32,
            Device::Cpu,
        )?;
        registry.finalize()?;
        Ok((registry, site))
    }

    fn projection(
        a: [[[f32; 2]; 1]; 2],
        b: [[[f32; 1]; 2]; 2],
        scales: [f32; 2],
    ) -> Result<LoraExpertProjectionWeights> {
        let device = Device::Cpu;
        LoraExpertProjectionWeights::new(
            Tensor::new(&a, &device)?,
            Tensor::new(&b, &device)?,
            Tensor::new(&scales, &device)?,
        )
    }

    fn adapter(
        site: &LoraExpertSiteHandle,
        gate_scale: [f32; 2],
        up_scale: [f32; 2],
        down_scale: [f32; 2],
    ) -> Result<LoraExpertWeights> {
        LoraExpertWeights::new(
            site,
            Some(projection(
                [[[1., 0.]], [[0., 1.]]],
                [[[1.], [2.]], [[3.], [4.]]],
                gate_scale,
            )?),
            Some(projection(
                [[[0., 1.]], [[1., 0.]]],
                [[[2.], [1.]], [[1.], [3.]]],
                up_scale,
            )?),
            Some(projection(
                [[[1., 0.]], [[0., 1.]]],
                [[[1.], [1.]], [[2.], [1.]]],
                down_scale,
            )?),
        )
    }

    #[test]
    fn packed_projection_validation_rejects_bad_shapes() -> Result<()> {
        let (_, site) = site()?;
        let device = Device::Cpu;
        let bad = LoraExpertProjectionWeights::new(
            Tensor::zeros((2, 1, 3), DType::F32, &device)?,
            Tensor::zeros((2, 2, 1), DType::F32, &device)?,
            Tensor::zeros(2, DType::F32, &device)?,
        )?;
        assert!(LoraExpertWeights::new(&site, Some(bad), None, None).is_err());
        assert!(LoraExpertProjectionWeights::new(
            Tensor::zeros((2, 1, 2), DType::F32, &device)?,
            Tensor::zeros((2, 2, 1), DType::F32, &device)?,
            Tensor::zeros(1, DType::F32, &device)?,
        )
        .is_err());
        assert!(LoraExpertProjectionWeights::new(
            Tensor::zeros((2, 1, 2), DType::F32, &device)?,
            Tensor::zeros((2, 2, 1), DType::F32, &device)?,
            Tensor::new(&[1f32, f32::NAN], &device)?,
        )
        .is_err());
        Ok(())
    }

    #[test]
    fn cpu_reference_supports_f16_and_bf16() -> Result<()> {
        for dtype in [DType::F16, DType::BF16] {
            let registry = LoraLayerRegistry::new();
            let site = registry.register_expert(
                LoraSiteKey::new(format!("experts.{dtype:?}")),
                LoraExpertSiteSpec::new(
                    1,
                    2,
                    2,
                    LoraExpertProjectionNames::new("gate", "up", "down"),
                    Shard::default(),
                    Shard::default(),
                )?,
                dtype,
                Device::Cpu,
            )?;
            registry.finalize()?;
            let projection = LoraExpertProjectionWeights::new(
                Tensor::new(&[[[1f32, 0.]]], &Device::Cpu)?.to_dtype(dtype)?,
                Tensor::new(&[[[1f32], [2.]]], &Device::Cpu)?.to_dtype(dtype)?,
                Tensor::new(&[1f32], &Device::Cpu)?,
            )?;
            let weights = LoraExpertWeights::new(&site, Some(projection), None, None)?;
            let mut execution = LoraExecution::new(registry.runtime_id(), vec![Some(1)]);
            execution.insert_expert(&site, 1, weights)?;
            let output = add_expert_delta_reference(
                &execution,
                &site,
                LoraExpertDelta::new(
                    LoraExpertProjection::Gate,
                    &Tensor::new(&[[1f32, 2.]], &Device::Cpu)?.to_dtype(dtype)?,
                    Tensor::zeros((1, 1, 2), dtype, &Device::Cpu)?,
                    &Tensor::new(&[[0u32]], &Device::Cpu)?,
                    LoraExpertInputMode::TokenRows,
                ),
            )?
            .to_dtype(DType::F32)?
            .to_vec3::<f32>()?;
            assert_eq!(output, vec![vec![vec![1., 2.]]]);
        }
        Ok(())
    }

    #[test]
    fn mixed_base_and_adapters_follow_token_and_expert_routes() -> Result<()> {
        let (registry, site) = site()?;
        let mut execution = LoraExecution::new(registry.runtime_id(), vec![Some(3), None, Some(7)]);
        execution.insert_expert(&site, 3, adapter(&site, [1., 0.], [1., 0.], [1., 0.])?)?;
        execution.insert_expert(&site, 7, adapter(&site, [0., 2.], [0., 0.5], [0., 1.5])?)?;

        let device = Device::Cpu;
        let input = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &device)?;
        let topk_ids = Tensor::new(&[[0u32, 1], [1, 0], [1, 0]], &device)?;
        let base_gate = Tensor::new(
            &[
                [[0.5f32, 1.], [1.5, 2.]],
                [[2.5, 3.], [3.5, 4.]],
                [[4.5, 5.], [5.5, 6.]],
            ],
            &device,
        )?;
        let base_up = Tensor::new(
            &[
                [[1f32, 1.5], [2., 2.5]],
                [[3., 3.5], [4., 4.5]],
                [[5., 5.5], [6., 6.5]],
            ],
            &device,
        )?;

        let gate = add_expert_delta_reference(
            &execution,
            &site,
            LoraExpertDelta::new(
                LoraExpertProjection::Gate,
                &input,
                base_gate.clone(),
                &topk_ids,
                LoraExpertInputMode::TokenRows,
            ),
        )?;
        let up = add_expert_delta_reference(
            &execution,
            &site,
            LoraExpertDelta::new(
                LoraExpertProjection::Up,
                &input,
                base_up.clone(),
                &topk_ids,
                LoraExpertInputMode::TokenRows,
            ),
        )?;

        let gate_values = gate.to_vec3::<f32>()?;
        let up_values = up.to_vec3::<f32>()?;
        assert_eq!(gate_values[1], base_gate.to_vec3::<f32>()?[1]);
        assert_eq!(up_values[1], base_up.to_vec3::<f32>()?[1]);
        assert_eq!(gate_values[0][0], vec![1.5, 3.]);
        assert_eq!(gate_values[0][1], vec![1.5, 2.]);
        assert_eq!(gate_values[2][0], vec![40.5, 53.]);
        assert_eq!(gate_values[2][1], vec![5.5, 6.]);
        assert_eq!(up_values[0][0], vec![5., 3.5]);
        assert_eq!(up_values[0][1], vec![2., 2.5]);
        assert_eq!(up_values[2][0], vec![7.5, 13.]);
        assert_eq!(up_values[2][1], vec![6., 6.5]);

        let down_input = gate.broadcast_mul(&up)?;
        let base_down = Tensor::new(
            &[
                [[1f32, 2.], [3., 4.]],
                [[5., 6.], [7., 8.]],
                [[9., 10.], [11., 12.]],
            ],
            &device,
        )?;
        let down = add_expert_delta_reference(
            &execution,
            &site,
            LoraExpertDelta::new(
                LoraExpertProjection::Down,
                &down_input,
                base_down.clone(),
                &topk_ids,
                LoraExpertInputMode::RoutedRows,
            ),
        )?;
        let down_values = down.to_vec3::<f32>()?;
        assert_eq!(down_values[1], base_down.to_vec3::<f32>()?[1]);
        assert_eq!(down_values[0][1], base_down.to_vec3::<f32>()?[0][1]);
        assert_eq!(down_values[2][1], base_down.to_vec3::<f32>()?[2][1]);

        let router_weights = Tensor::new(&[[0.75f32, 0.25], [0.4, 0.6], [0.2, 0.8]], &device)?;
        let reduced = down.broadcast_mul(&router_weights.unsqueeze(2)?)?.sum(1)?;
        let reduced_values = reduced.to_vec2::<f32>()?;
        let expected = down_values
            .iter()
            .zip(router_weights.to_vec2::<f32>()?)
            .map(|(routes, weights)| {
                (0..2)
                    .map(|feature| {
                        routes[0][feature] * weights[0] + routes[1][feature] * weights[1]
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(reduced_values, expected);
        Ok(())
    }

    #[test]
    fn routed_weights_scale_down_delta_before_it_is_added() -> Result<()> {
        let (registry, site) = site()?;
        let mut execution = LoraExecution::new(registry.runtime_id(), vec![Some(3)]);
        execution.insert_expert(&site, 3, adapter(&site, [0., 0.], [0., 0.], [1., 1.])?)?;

        let device = Device::Cpu;
        let input = Tensor::new(&[[[2f32, 0.], [0., 3.]]], &device)?;
        let topk_ids = Tensor::new(&[[0u32, 1]], &device)?;
        let routed_weights = Tensor::new(&[[0.25f32, 0.75]], &device)?;
        let delta = LoraExpertDelta::new(
            LoraExpertProjection::Down,
            &input,
            Tensor::zeros((1, 2, 2), DType::F32, &device)?,
            &topk_ids,
            LoraExpertInputMode::RoutedRows,
        )
        .with_routed_weights(&routed_weights);
        let output = add_expert_delta_reference(&execution, &site, delta)?;

        assert_eq!(
            output.to_vec3::<f32>()?,
            vec![vec![vec![0.5, 0.5], vec![4.5, 2.25]]]
        );
        Ok(())
    }
}
