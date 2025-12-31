struct DecoderLayer {
    self_attn: Attention,
    mlp: MoeOrMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        real_device: Device,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb,
            cfg,
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            mapper,
            layer_idx,
            loading_isq,
            paged_attn,
            comm,
        )?;

        let mlp = if !cfg.mlp_only_layers.contains(&layer_idx)
            && (cfg.num_experts > 0 && (layer_idx + 1) % cfg.decoder_sparse_step == 0)
        {
            let vb = mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq);
            let layer_device = mapper
                .device_for(layer_idx, false)
                .cloned()
                .unwrap_or(real_device);

            MoeOrMlp::Moe(MoeMlp::new(cfg, vb, layer_device, comm, loading_isq)?)
        } else {
            MoeOrMlp::Mlp(Mlp::new(
                cfg,
                mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq),
                comm,
                cfg.intermediate_size,
            )?)
        };
        let input_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offsets,
            kv_cache,
            metadata,
            flash_params,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.mlp.forward(
            &xs.apply(&self.post_attention_layernorm)?,
            flash_params.causal,
        )?;
        residual + xs
    }
}
