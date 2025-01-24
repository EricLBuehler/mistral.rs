#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

mod config;
mod inputs_processor;
mod vision;

use std::any::Any;

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::VarBuilder;
pub use config::Idefics3Config;
pub use inputs_processor::Idefics3Processor;
use vision::{Idefics3Connector, Idefics3VisionTransformer};

use crate::{
    amoe::{AnyMoeBaseModelMixin, MlpLayer},
    device_map::DeviceMapper,
    models::llama::Llama,
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, NormalLoadingMetadata, NormalModel, VisionModel,
    },
    utils::unvarbuilder::UnVarBuilder,
    AnyMoeConfig, AnyMoeExpertType,
};

pub struct Idefics3Model {
    text_model: Llama,
    connector: Idefics3Connector,
    vision: Idefics3VisionTransformer,
    config: Idefics3Config,
    dtype: DType,
}

impl Idefics3Model {
    pub fn new(
        cfg: &Idefics3Config,
        vb: VarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let connector = Idefics3Connector::new(
            cfg,
            vb_m.pp("connector")
                .set_dtype(DType::F32)
                .set_device(normal_loading_metadata.real_device.clone()),
        )?;
        let vision = Idefics3VisionTransformer::new(
            &cfg.vision_config,
            vb_m.pp("vision_model")
                .set_dtype(DType::F32)
                .set_device(normal_loading_metadata.real_device.clone()),
        )?;
        let text_model = Llama::new_inner(
            &cfg.text_config,
            vb_m.pp("text_model"),
            vb.pp("lm_head"),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        Ok(Self {
            text_model,
            connector,
            vision,
            config: cfg.clone(),
            dtype: vb.dtype(),
        })
    }

    fn inputs_merger(
        &self,
        input_ids: &Tensor,
        input_embeds: &Tensor,
        image_hidden_states: &Tensor,
    ) -> Result<Tensor> {
        // Docs copied from Transformers impl
        /*
        This method aims at merging the token embeddings with the image hidden states into one single sequence of vectors that are fed to the transformer LM.
        The merging happens as follows:
        - The text token sequence is: `tok_1 tok_2 tok_3 <fake_token_around_image> <image> <image> ... <image> <fake_token_around_image> tok_4`.
        - We get the image hidden states for the image through the vision encoder (and potentially the perceiver), and that hidden state is then projected into the text embedding space.
        We thus have a sequence of image hidden states of size (1, image_seq_len, hidden_dim), where 1 is for batch_size of 1 image and hidden_dim is the hidden_dim of the LM transformer.
        - The merging happens so that we obtain the following sequence: `vector_tok_1 vector_tok_2 vector_tok_3 vector_fake_tok_around_image {sequence of image_seq_len image hidden states} vector_fake_toke_around_image vector_tok_4`. That sequence is fed to the LM.
        - To fit the format of that sequence, `input_ids`, `input_embeds`, `attention_mask` are all 3 adapted to insert the image hidden states.
        */
        let (_, _, vision_hidden_size) = image_hidden_states.dims3()?;
        let bs = input_ids.dim(0)?;
        let special_image_token_mask = input_ids.eq(self.config.image_token_id as f64)?;
        let mut new_inputs_embeds = input_embeds.clone();
        let reshaped_image_hidden_states =
            image_hidden_states.reshape((bs, (), vision_hidden_size))?;
        assert_eq!(input_embeds.dim(0)?, 1);
        assert_eq!(reshaped_image_hidden_states.dim(0)?, 1);
        let special_image_token_mask = special_image_token_mask.i(0)?.to_vec1::<u8>()?;
        let mut image_hidden_state_i = 0;
        for (i, v) in special_image_token_mask.iter().enumerate() {
            if *v != 0 {
                new_inputs_embeds = new_inputs_embeds.slice_assign(
                    &[&.., &i, &..],
                    &reshaped_image_hidden_states
                        .i((.., image_hidden_state_i, ..))?
                        .unsqueeze(1)?,
                )?;
                image_hidden_state_i += 1;
            }
        }
        Ok(new_inputs_embeds)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        pixel_attention_mask: Option<Tensor>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &mut PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let input_embeds = if let Some(pixel_values) = pixel_values {
            // == START VISUAL INPUTS INTEGRATION ==
            let (batch_size, num_images, _, _, _) = pixel_values.dims5()?;
            let mut s = vec![batch_size * num_images];
            s.extend(pixel_values.dims()[2..].to_vec());
            let pixel_values = pixel_values.reshape(s)?;

            // Remove padding images which are full of 0s
            let nb_values_per_image = pixel_values.dims()[1..].iter().product::<usize>();
            let real_images_inds = pixel_values
                .eq(0.0f64)?
                .sum(vec![
                    pixel_values.dims().len() - 1,
                    pixel_values.dims().len() - 2,
                    pixel_values.dims().len() - 3,
                ])?
                .ne(nb_values_per_image as f64)?;
            let mut batches = Vec::new();
            for (batch, use_it) in pixel_values
                .chunk(pixel_values.dim(0)?, 0)?
                .iter()
                .zip(real_images_inds.chunk(real_images_inds.dim(0)?, 0)?)
            {
                let use_it = use_it.squeeze(0)?.to_scalar::<u8>()? != 0;
                if use_it {
                    batches.push(batch.clone());
                }
            }
            let pixel_values = Tensor::cat(&batches, 0)?;

            // Vision attention mask
            let pixel_attention_mask = if let Some(pixel_attention_mask) = pixel_attention_mask {
                let pixel_attention_mask = pixel_attention_mask.reshape((
                    batch_size * num_images,
                    pixel_attention_mask.dims()[2],
                    pixel_attention_mask.dims()[3],
                ))?;
                let mut batches = Vec::new();
                for (batch, use_it) in pixel_attention_mask
                    .chunk(pixel_attention_mask.dim(0)?, 0)?
                    .iter()
                    .zip(real_images_inds.chunk(real_images_inds.dim(0)?, 0)?)
                {
                    let use_it = use_it.squeeze(0)?.to_scalar::<u8>()? != 0;
                    if use_it {
                        batches.push(batch.clone());
                    }
                }
                Tensor::cat(&batches, 0)?
            } else {
                Tensor::ones(
                    (
                        pixel_values.dims()[0],
                        pixel_values.dims()[2],
                        pixel_values.dims()[3],
                    ),
                    DType::U8,
                    pixel_values.device(),
                )?
            };

            let patch_size = self.config.vision_config.patch_size;
            let patches_subgrid = pixel_attention_mask.unfold(1, patch_size, patch_size)?;
            let patches_subgrid = patches_subgrid.unfold(2, patch_size, patch_size)?;

            let patch_attention_mask = patches_subgrid
                .sum((D::Minus1, D::Minus2))?
                .gt(0.0)?
                .to_dtype(DType::U8)?;

            let pixel_values = pixel_values.to_dtype(self.dtype)?;

            // Get seq from vision encoder
            let image_hidden_states = self.vision.forward(
                &pixel_values.to_dtype(DType::F32)?,
                Some(&patch_attention_mask),
            )?;

            // Modality proj and perceiver resampling
            let image_hidden_states = self.connector.forward(&image_hidden_states)?;

            if self.text_model.cache().normal().0[0].current_seq_len() == 0 {
                self.inputs_merger(
                    input_ids,
                    &self
                        .text_model
                        .get_input_embeddings(input_ids)?
                        .to_dtype(DType::F32)?,
                    &image_hidden_states,
                )?
                .to_dtype(self.dtype)?
            } else {
                candle_core::bail!("Pixel values were specified for a non-prompt.")
            }
        } else {
            self.text_model.get_input_embeddings(input_ids)?
        };

        self.text_model.forward_embeds(
            input_ids,
            input_embeds,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
    }
}

impl IsqModel for Idefics3Model {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(
            &mut std::sync::Arc<dyn mistralrs_quant::QuantMethod>,
            Option<usize>,
        )>,
        &dyn DeviceMapper,
    ) {
        self.text_model.get_layers()
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        let uvb_m = uvb.pp("model");
        uvb_m
            .pp("connector")
            .pp("modality_projection")
            .pp("proj")
            .add(&self.connector.modality_projection.proj);
        uvb.extend(self.text_model.residual_tensors_m(uvb_m.pp("text_model")));
        uvb_m
            .pp("vision_model")
            .extend(self.vision.residual_tensors());

        uvb.to_safetensors()
    }
}

// AnyMoE is forwarded to the base model
impl AnyMoeBaseModelMixin for Idefics3Model {
    fn get_mlps(&self) -> Vec<&dyn MlpLayer> {
        self.text_model.get_mlps()
    }
    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        self.text_model.get_mlps_mut()
    }
    fn create_anymoe_layers(
        &mut self,
        additional_vbs: Vec<VarBuilder>,
        config: AnyMoeConfig,
        (prefix, mlp): (String, String),
        layers: Vec<usize>,
        expert_type: AnyMoeExpertType,
        gate_vb: Option<VarBuilder>,
    ) -> Result<()> {
        self.text_model.create_anymoe_layers(
            additional_vbs,
            config,
            (prefix, mlp),
            layers,
            expert_type,
            gate_vb,
        )
    }
    fn amoe_supported(&self) -> bool {
        true
    }
}

impl VisionModel for Idefics3Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _: Vec<usize>, // Ignore, it is for phi3
        model_specific_args: Box<dyn Any>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &mut PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> candle_core::Result<Tensor> {
        let pixel_attention_mask: Option<Tensor> = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `Option<Tensor>`");
        self.forward_inner(
            input_ids,
            pixel_values,
            seqlen_offsets,
            context_lens,
            pixel_attention_mask,
            metadata,
            flash_params,
        )
    }
    fn cache(&self) -> &EitherCache {
        self.text_model.cache()
    }
    fn cache_mut(&mut self) -> &mut EitherCache {
        self.text_model.cache_mut()
    }
    fn device(&self) -> &Device {
        self.text_model.device()
    }
    fn max_seq_len(&self) -> usize {
        self.text_model.max_seq_len()
    }
    fn has_conv2d(&self) -> bool {
        true
    }
    fn config(&self) -> &ModelConfigMetadata {
        self.text_model.config()
    }
    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn Any> {
        let args: Option<Tensor> = None;
        Box::new(args)
    }
}
