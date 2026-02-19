#[doc(hidden)]
#[macro_export]
macro_rules! api_dir_list {
    ($api:expr, $model_id:expr, $should_panic:expr) => {{
        let model_path = std::path::Path::new($model_id);
        let files = $crate::pipeline::hf::list_repo_files(&$api, model_path, $should_panic)
            .map_err(candle_core::Error::msg)?;
        files.into_iter()
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! api_get_file {
    ($api:expr, $file:expr, $model_id:expr) => {{
        let model_path = std::path::Path::new($model_id);
        $crate::pipeline::hf::get_file(&$api, model_path, $file).map_err(candle_core::Error::msg)?
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! get_paths {
    (
        $path_name:ident,
        $token_source:expr,
        $revision:expr,
        $this:expr,
        $quantized_model_id:expr,
        $quantized_filename:expr,
        $silent:expr,
        $loading_uqff:expr
    ) => {{
        let api = {
            use $crate::GLOBAL_HF_CACHE;
            let cache = GLOBAL_HF_CACHE.get().cloned().unwrap_or_default();
            let mut api = ApiBuilder::from_cache(cache)
                .with_progress(!$silent)
                .with_token(get_token($token_source)?);
            if let Some(cache_dir) = $crate::hf_hub_cache_dir() {
                api = api.with_cache_dir(cache_dir);
            }
            api.build()?
        };
        let revision = $revision.unwrap_or("main".to_string());
        let api = api.repo(Repo::with_revision(
            $this.model_id.clone(),
            RepoType::Model,
            revision.clone(),
        ));
        let model_id = std::path::Path::new(&$this.model_id);
        let dir_list = $crate::api_dir_list!(api, model_id, false).collect::<Vec<_>>();
        let tokenizer_filename = if let Some(ref p) = $this.tokenizer_json {
            info!("Using tokenizer.json at `{p}`");
            PathBuf::from_str(p)?
        } else if dir_list.contains(&"tokenizer.json".to_string()) {
            info!("Loading `tokenizer.json` at `{}`", $this.model_id);
            $crate::api_get_file!(api, "tokenizer.json", model_id)
        } else if dir_list.contains(&"tekken.json".to_string()) {
            info!(
                "Loading `tekken.json` (Mistral tokenizer) at `{}`",
                $this.model_id
            );
            $crate::api_get_file!(api, "tekken.json", model_id)
        } else {
            info!("Loading `tokenizer.json` at `{}`", $this.model_id);
            $crate::api_get_file!(api, "tokenizer.json", model_id)
        };
        let config_filename = if dir_list.contains(&"config.json".to_string()) {
            info!("Loading `config.json` at `{}`", $this.model_id);
            $crate::api_get_file!(api, "config.json", model_id)
        } else if dir_list.contains(&"params.json".to_string()) {
            info!(
                "Loading `params.json` (Mistral config) at `{}`",
                $this.model_id
            );
            $crate::api_get_file!(api, "params.json", model_id)
        } else {
            info!("Loading `config.json` at `{}`", $this.model_id);
            $crate::api_get_file!(api, "config.json", model_id)
        };
        let filenames = get_model_paths(
            revision.clone(),
            &$token_source,
            $quantized_model_id.as_ref(),
            $quantized_filename.as_ref(),
            &api,
            &model_id,
            $loading_uqff,
        )?;
        let adapter_paths = get_xlora_paths(
            $this.model_id.clone(),
            $this.xlora_model_id.as_ref(),
            $this.lora_adapter_ids.as_ref(),
            &$token_source,
            revision.clone(),
            $this.xlora_order.as_ref(),
        )?;

        let gen_conf = if dir_list.contains(&"generation_config.json".to_string()) {
            info!("Loading `generation_config.json` at `{}`", $this.model_id);
            Some($crate::api_get_file!(
                api,
                "generation_config.json",
                model_id
            ))
        } else {
            None
        };
        let preprocessor_config = if dir_list.contains(&"preprocessor_config.json".to_string()) {
            info!("Loading `preprocessor_config.json` at `{}`", $this.model_id);
            Some($crate::api_get_file!(
                api,
                "preprocessor_config.json",
                model_id
            ))
        } else {
            None
        };
        let processor_config = if dir_list.contains(&"processor_config.json".to_string()) {
            info!("Loading `processor_config.json` at `{}`", $this.model_id);
            Some($crate::api_get_file!(
                api,
                "processor_config.json",
                model_id
            ))
        } else {
            None
        };
        let template_filename = if let Some(ref p) = $this.chat_template {
            info!("Using chat template file at `{p}`");
            Some(PathBuf::from_str(p)?)
        } else if dir_list.contains(&"chat_template.jinja".to_string()) {
            info!("Loading `chat_template.jinja` at `{}`", $this.model_id);
            Some($crate::api_get_file!(api, "chat_template.jinja", model_id))
        } else if dir_list.contains(&"tokenizer_config.json".to_string()) {
            info!("Loading `tokenizer_config.json` at `{}`", $this.model_id);
            Some($crate::api_get_file!(
                api,
                "tokenizer_config.json",
                model_id
            ))
        } else {
            info!(
                "No chat template or `tokenizer_config.json` found at `{}`",
                $this.model_id
            );
            None
        };
        let chat_template_json_filename = if dir_list.contains(&"chat_template.json".to_string()) {
            info!("Loading `chat_template.json` at `{}`", $this.model_id);
            Some($crate::api_get_file!(api, "chat_template.json", model_id))
        } else {
            None
        };
        Ok(Box::new($path_name {
            tokenizer_filename,
            config_filename,
            filenames,
            adapter_paths,
            template_filename,
            gen_conf,
            preprocessor_config,
            processor_config,
            chat_template_json_filename,
        }))
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! get_embedding_paths {
    (
        $path_name:ident,
        $token_source:expr,
        $revision:expr,
        $this:expr,
        $quantized_model_id:expr,
        $quantized_filename:expr,
        $silent:expr,
        $loading_uqff:expr
    ) => {{
        let api = {
            use $crate::GLOBAL_HF_CACHE;
            let cache = GLOBAL_HF_CACHE.get().cloned().unwrap_or_default();
            let mut api = ApiBuilder::from_cache(cache)
                .with_progress(!$silent)
                .with_token(get_token($token_source)?);
            if let Some(cache_dir) = $crate::hf_hub_cache_dir() {
                api = api.with_cache_dir(cache_dir);
            }
            api.build()?
        };
        let revision = $revision.unwrap_or("main".to_string());
        let api = api.repo(Repo::with_revision(
            $this.model_id.clone(),
            RepoType::Model,
            revision.clone(),
        ));
        let model_id = std::path::Path::new(&$this.model_id);
        let emb_dir_list = $crate::api_dir_list!(api, model_id, false).collect::<Vec<_>>();
        let tokenizer_filename = if let Some(ref p) = $this.tokenizer_json {
            info!("Using tokenizer.json at `{p}`");
            PathBuf::from_str(p)?
        } else if emb_dir_list.contains(&"tokenizer.json".to_string()) {
            info!("Loading `tokenizer.json` at `{}`", $this.model_id);
            $crate::api_get_file!(api, "tokenizer.json", model_id)
        } else if emb_dir_list.contains(&"tekken.json".to_string()) {
            info!(
                "Loading `tekken.json` (Mistral tokenizer) at `{}`",
                $this.model_id
            );
            $crate::api_get_file!(api, "tekken.json", model_id)
        } else {
            info!("Loading `tokenizer.json` at `{}`", $this.model_id);
            $crate::api_get_file!(api, "tokenizer.json", model_id)
        };
        let config_filename = if emb_dir_list.contains(&"config.json".to_string()) {
            info!("Loading `config.json` at `{}`", $this.model_id);
            $crate::api_get_file!(api, "config.json", model_id)
        } else if emb_dir_list.contains(&"params.json".to_string()) {
            info!(
                "Loading `params.json` (Mistral config) at `{}`",
                $this.model_id
            );
            $crate::api_get_file!(api, "params.json", model_id)
        } else {
            info!("Loading `config.json` at `{}`", $this.model_id);
            $crate::api_get_file!(api, "config.json", model_id)
        };
        let filenames = get_model_paths(
            revision.clone(),
            &$token_source,
            $quantized_model_id.as_ref(),
            $quantized_filename.as_ref(),
            &api,
            &model_id,
            $loading_uqff,
        )?;
        let adapter_paths = get_xlora_paths(
            $this.model_id.clone(),
            None, // no xlora
            $this.lora_adapter_ids.as_ref(),
            &$token_source,
            revision.clone(),
            None, // no xlora
        )?;

        let mut parsed_modules = Vec::new();
        let is_local = std::path::Path::new(&$this.model_id).exists();
        let modules_path = if is_local {
            model_id.join("modules.json")
        } else {
            $crate::api_get_file!(api, "modules.json", model_id)
        };

        if modules_path.exists() {
            let modules: Vec<$crate::pipeline::EmbeddingModule> =
                serde_json::from_str(&std::fs::read_to_string(&modules_path)?)?;
            for module in modules {
                match module.ty {
                    $crate::pipeline::EmbeddingModuleType::Transformer => {
                        parsed_modules.push($crate::pipeline::EmbeddingModulePaths::Transformer {
                            path: module.path.clone(),
                        });
                    }
                    $crate::pipeline::EmbeddingModuleType::Pooling => {
                        parsed_modules.push($crate::pipeline::EmbeddingModulePaths::Pooling {
                            path: module.path.clone(),
                            config: $crate::api_get_file!(
                                api,
                                &format!("{}/config.json", module.path),
                                model_id
                            ),
                        });
                    }
                    $crate::pipeline::EmbeddingModuleType::Dense => {
                        parsed_modules.push($crate::pipeline::EmbeddingModulePaths::Dense {
                            path: module.path.clone(),
                            config: $crate::api_get_file!(
                                api,
                                &format!("{}/config.json", module.path),
                                model_id
                            ),
                            model: $crate::api_get_file!(
                                api,
                                &format!("{}/model.safetensors", module.path),
                                model_id
                            ),
                        });
                    }
                    $crate::pipeline::EmbeddingModuleType::Normalize => {
                        parsed_modules.push($crate::pipeline::EmbeddingModulePaths::Normalize {
                            path: module.path.clone(),
                        });
                    }
                }
            }
        }

        Ok(Box::new($path_name {
            tokenizer_filename,
            config_filename,
            filenames,
            adapter_paths,
            modules: parsed_modules,
        }))
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! get_uqff_paths {
    ($from_uqff:expr, $this:expr, $silent:expr) => {{
        let api = {
            use $crate::GLOBAL_HF_CACHE;
            let cache = GLOBAL_HF_CACHE.get().cloned().unwrap_or_default();
            let mut api = ApiBuilder::from_cache(cache)
                .with_progress(!$silent)
                .with_token(get_token(
                    &$this
                        .token_source
                        .read()
                        .expect("Failed to read token source")
                        .clone()
                        .unwrap_or(TokenSource::None),
                )?);
            if let Some(cache_dir) = $crate::hf_hub_cache_dir() {
                api = api.with_cache_dir(cache_dir);
            }
            api.build()?
        };
        let revision = $this
            .revision
            .read()
            .expect("Failed to read revision")
            .clone()
            .unwrap_or("main".to_string());
        let api = api.repo(Repo::with_revision(
            $this.model_id.to_string(),
            RepoType::Model,
            revision.clone(),
        ));

        // Auto-discover UQFF shard siblings
        let available_files =
            $crate::pipeline::hf::list_repo_files(&api, Path::new(&$this.model_id), false)
                .unwrap_or_default();

        let input_files: Vec<String> = $from_uqff.iter().map(|f| f.display().to_string()).collect();
        let input_count = input_files.len();

        let mut expanded_files: Vec<String> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for file_str in &input_files {
            let expanded = $crate::pipeline::isq::expand_uqff_shards(file_str, &available_files);
            for f in expanded {
                if seen.insert(f.clone()) {
                    expanded_files.push(f);
                }
            }
        }

        if expanded_files.len() > input_count {
            tracing::info!(
                "Auto-discovered {} UQFF shard files (from {} specified)",
                expanded_files.len(),
                input_count
            );
        }

        let mut files = Vec::new();
        for file in &expanded_files {
            files.push($crate::api_get_file!(api, file, Path::new(&$this.model_id)));
        }
        files
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! get_paths_gguf {
    (
        $path_name:ident,
        $token_source:expr,
        $revision:expr,
        $this:expr,
        $quantized_model_id:expr,
        $quantized_filenames:expr,
        $silent:expr
    ) => {{
        let api = {
            use $crate::GLOBAL_HF_CACHE;
            let cache = GLOBAL_HF_CACHE.get().cloned().unwrap_or_default();
            let mut api = ApiBuilder::from_cache(cache)
                .with_progress(!$silent)
                .with_token(get_token($token_source)?);
            if let Some(cache_dir) = $crate::hf_hub_cache_dir() {
                api = api.with_cache_dir(cache_dir);
            }
            api.build()?
        };
        let revision = $revision.unwrap_or("main".to_string());
        let this_model_id = $this.model_id.clone().unwrap_or($this.quantized_model_id.clone());
        let api = api.repo(Repo::with_revision(
            this_model_id.clone(),
            RepoType::Model,
            revision.clone(),
        ));
        let model_id = std::path::Path::new(&this_model_id);

        let dir_list = $crate::api_dir_list!(api, model_id, false)
            .collect::<Vec<_>>();

        let chat_template = if let Some(ref p) = $this.chat_template {
            if p.ends_with(".json") || p.ends_with(".jinja") {
                info!("Using chat template file at `{p}`");
                Some(PathBuf::from_str(p)?)
            } else {
                panic!("Specified chat template file must end with .json or .jinja");
            }
        } else {
            if $this.model_id.is_none() {
                None
            } else if dir_list.contains(&"chat_template.jinja".to_string()) {
                info!("Loading `chat_template.jinja` at `{}`", this_model_id);
                Some($crate::api_get_file!(
                    api,
                    "chat_template.jinja",
                    model_id
                ))
            } else if dir_list.contains(&"tokenizer_config.json".to_string()) {
                info!("Loading `tokenizer_config.json` at `{}` because no chat template file was specified.", this_model_id);
                let res = $crate::api_get_file!(
                    api,
                    "tokenizer_config.json",
                    model_id
                );
                Some(res)
            } else {
                info!("No chat template or `tokenizer_config.json` found at `{}`", this_model_id);
                None
            }
        };

        let filenames = get_model_paths(
            revision.clone(),
            &$token_source,
            Some(&$quantized_model_id),
            Some(&$quantized_filenames),
            &api,
            &model_id,
            false, // Never loading UQFF
        )?;

        info!("GGUF file(s) {:?}", filenames);
        let adapter_paths = get_xlora_paths(
            this_model_id.clone(),
            $this.xlora_model_id.as_ref(),
            $this.lora_adapter_ids.as_ref(),
            &$token_source,
            revision.clone(),
            $this.xlora_order.as_ref(),
        )?;

        let gen_conf = if dir_list.contains(&"generation_config.json".to_string()) {
            info!("Loading `generation_config.json` at `{}`", this_model_id);
            Some($crate::api_get_file!(
                api,
                "generation_config.json",
                model_id
            ))
        } else {
            None
        };

        let preprocessor_config = if dir_list.contains(&"preprocessor_config.json".to_string())
        {
            info!("Loading `preprocessor_config.json` at `{}`", this_model_id);
            Some($crate::api_get_file!(
                api,
                "preprocessor_config.json",
                model_id
            ))
        } else {
            None
        };

        let processor_config = if dir_list.contains(&"processor_config.json".to_string()) {
            info!("Loading `processor_config.json` at `{}`", this_model_id);
            Some($crate::api_get_file!(
                api,
                "processor_config.json",
                model_id
            ))
        } else {
            None
        };

        let tokenizer_filename = if $this.model_id.is_some() && dir_list.contains(&"tokenizer.json".to_string()) {
            info!("Loading `tokenizer.json` at `{}`", this_model_id);
            $crate::api_get_file!(api, "tokenizer.json", model_id)
        } else {
            PathBuf::from_str("")?
        };

        let chat_template_json_filename = if dir_list.contains(&"chat_template.json".to_string()) {
            info!("Loading `chat_template.json` at `{}`", this_model_id);
            Some($crate::api_get_file!(
                api,
                "chat_template.json",
                model_id
            ))
        } else {
            None
        };

        Ok(Box::new($path_name {
            tokenizer_filename,
            config_filename: PathBuf::from_str("")?,
            filenames,
            adapter_paths,
            template_filename: chat_template,
            gen_conf,
            preprocessor_config,
            processor_config,
            chat_template_json_filename,
        }))
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! normal_model_loader {
    (
        $paths:expr,
        $dtype:expr,
        $device:expr,
        $layer_devices:expr,
        $config:expr,
        $loader:expr,
        $silent:expr,
        $mapper:expr,
        $loading_isq:expr,
        $loading_uqff:expr,
        $real_device:expr,
        $attention_mechanism:expr,
        $is_moqe:expr,
        $multi_progress:expr,
        $matformer_config:expr,
    ) => {{
        let regexes = if $loading_isq && $loading_uqff {
            // Dummy weights for the layers which will be overwritten...
            Some(std::sync::Arc::new(if $is_moqe {
                $loader.isq_layer_regexes_moqe(&$config)?
            } else {
                $loader.isq_layer_regexes(&$config)?
            }))
        } else {
            None
        };
        let get_device_for_tensor =
            $loader.get_device_for_tensor(&$config, &*$mapper, $loading_isq)?;

        let vb = from_mmaped_safetensors(
            $paths.get_weight_filenames().to_vec(),
            Vec::new(),
            $dtype,
            $device,
            $layer_devices,
            $silent,
            regexes,
            |_| true, // Will be overwritten...
            get_device_for_tensor,
        )?;

        $loader.load(
            &$config,
            vb,
            $crate::pipeline::NormalLoadingMetadata {
                mapper: $mapper,
                loading_isq: $loading_isq,
                real_device: $real_device,
                multi_progress: $multi_progress,
                matformer_slicing_config: $matformer_config,
            },
            $attention_mechanism,
        )?
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! normal_model_loader_sharded {
    (
        $vb:expr,
        $config:expr,
        $loader:expr,
        $mapper:expr,
        $loading_isq:expr,
        $real_device:expr,
        $attention_mechanism:expr,
        $multi_progress:expr,
        $matformer_config:expr,
    ) => {{
        $loader.load(
            &$config,
            $vb,
            $crate::pipeline::NormalLoadingMetadata {
                mapper: $mapper,
                loading_isq: $loading_isq,
                real_device: $real_device,
                multi_progress: $multi_progress,
                matformer_slicing_config: $matformer_config,
            },
            $attention_mechanism,
        )?
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! vision_normal_model_loader {
    (
        $paths:expr,
        $dtype:expr,
        $device:expr,
        $layer_devices:expr,
        $config:expr,
        $loader:expr,
        $silent:expr,
        $mapper:expr,
        $loading_isq:expr,
        $loading_uqff:expr,
        $real_device:expr,
        $attention_mechanism:expr,
        $is_moqe:expr,
        $multi_progress:expr,
        $matformer_config:expr,
    ) => {{
        let regexes = if $loading_isq && $loading_uqff {
            // Dummy weights for the layers which will be overwritten...
            Some(std::sync::Arc::new(if $is_moqe {
                $loader.isq_layer_regexes_moqe(&$config)?
            } else {
                $loader.isq_layer_regexes(&$config)?
            }))
        } else {
            None
        };
        let get_device_for_tensor =
            $loader.get_device_for_tensor(&$config, &*$mapper, $loading_isq)?;

        let vb = from_mmaped_safetensors(
            $paths.get_weight_filenames().to_vec(),
            Vec::new(),
            $dtype,
            $device,
            $layer_devices,
            $silent,
            regexes,
            |_| true, // Will be overwritten...
            get_device_for_tensor,
        )?;

        $loader.load(
            &$config,
            vb,
            $crate::pipeline::NormalLoadingMetadata {
                mapper: $mapper,
                loading_isq: $loading_isq,
                real_device: $real_device,
                multi_progress: $multi_progress,
                matformer_slicing_config: $matformer_config,
            },
            $attention_mechanism,
        )?
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! vision_normal_model_loader_sharded {
    (
        $vb:expr,
        $config:expr,
        $loader:expr,
        $mapper:expr,
        $loading_isq:expr,
        $real_device:expr,
        $attention_mechanism:expr,
        $multi_progress:expr,
        $matformer_config:expr,
    ) => {{
        $loader.load(
            &$config,
            $vb,
            $crate::pipeline::NormalLoadingMetadata {
                mapper: $mapper,
                loading_isq: $loading_isq,
                real_device: $real_device,
                multi_progress: $multi_progress,
                matformer_slicing_config: $matformer_config,
            },
            $attention_mechanism,
        )?
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! embedding_normal_model_loader {
    (
        $paths:expr,
        $dtype:expr,
        $device:expr,
        $layer_devices:expr,
        $config:expr,
        $loader:expr,
        $silent:expr,
        $mapper:expr,
        $loading_isq:expr,
        $loading_uqff:expr,
        $real_device:expr,
        $attention_mechanism:expr,
        $multi_progress:expr,
    ) => {{
        let regexes = if $loading_isq && $loading_uqff {
            // Dummy weights for the layers which will be overwritten...
            Some(std::sync::Arc::new($loader.isq_layer_regexes(&$config)?))
        } else {
            None
        };
        let get_device_for_tensor =
            $loader.get_device_for_tensor(&$config, &*$mapper, $loading_isq)?;

        let vb = from_mmaped_safetensors(
            $paths.get_weight_filenames().to_vec(),
            Vec::new(),
            $dtype,
            $device,
            $layer_devices,
            $silent,
            regexes,
            |_| true, // Will be overwritten...
            get_device_for_tensor,
        )?;

        $loader.load(
            &$config,
            vb,
            $crate::pipeline::NormalLoadingMetadata {
                mapper: $mapper,
                loading_isq: $loading_isq,
                real_device: $real_device,
                multi_progress: $multi_progress,
                matformer_slicing_config: None,
            },
            $attention_mechanism,
        )?
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! embedding_normal_model_loader_sharded {
    (
        $vb:expr,
        $config:expr,
        $loader:expr,
        $mapper:expr,
        $loading_isq:expr,
        $real_device:expr,
        $attention_mechanism:expr,
        $multi_progress:expr,
    ) => {{
        $loader.load(
            &$config,
            $vb,
            $crate::pipeline::NormalLoadingMetadata {
                mapper: $mapper,
                loading_isq: $loading_isq,
                real_device: $real_device,
                multi_progress: $multi_progress,
                matformer_slicing_config: None,
            },
            $attention_mechanism,
        )?
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! xlora_model_loader {
    (
        $paths:expr,
        $dtype:expr,
        $device:expr,
        $layer_devices:expr,
        $config:expr,
        $loader:expr,
        $silent:expr,
        $mapper:expr,
        $loading_isq:expr,
        $real_device:expr,
        $multi_progress:expr,
        $matformer_config:expr,
    ) => {{
        // TODO: remove lora_preload_adapter_info
        let $crate::pipeline::AdapterPaths::XLora {
            adapter_configs,
            adapter_safetensors,
            classifier_path,
            xlora_order,
            xlora_config,
            lora_preload_adapter_info: _,
        } = $paths.get_adapter_paths()
        else {
            unreachable!()
        };

        let mut safetensors_paths = $paths.get_weight_filenames().iter().collect::<Vec<_>>();
        safetensors_paths.push(classifier_path.as_ref().unwrap());
        let get_device_for_tensor =
            $loader.get_device_for_tensor(&$config, &*$mapper, $loading_isq)?;

        let vb = from_mmaped_safetensors(
            safetensors_paths
                .iter()
                .map(|x| (*x).to_owned())
                .collect::<Vec<_>>(),
            adapter_safetensors
                .as_ref()
                .unwrap()
                .iter()
                .map(|(_, x)| (*x).to_owned())
                .collect::<Vec<_>>(),
            $dtype,
            $device,
            $layer_devices,
            $silent,
            None,
            |_| true,
            get_device_for_tensor,
        )?;

        $loader.load_xlora(
            &$config,
            vb,
            adapter_configs.as_ref().unwrap(),
            Some(xlora_config.as_ref().unwrap().clone()),
            xlora_order.as_ref().unwrap().clone(),
            $crate::pipeline::NormalLoadingMetadata {
                mapper: $mapper,
                loading_isq: $loading_isq,
                real_device: $real_device,
                multi_progress: $multi_progress,
                matformer_slicing_config: $matformer_config,
            },
            &None,
        )?
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! lora_model_loader {
    (
        $paths:expr,
        $dtype:expr,
        $device:expr,
        $layer_devices:expr,
        $config:expr,
        $loader:expr,
        $silent:expr,
        $mapper:expr,
        $loading_isq:expr,
        $loading_uqff:expr,
        $real_device:expr,
        $attention_mechanism:expr,
        $is_moqe:expr,
        $multi_progress:expr,
        $matformer_config:expr,
    ) => {{
        let $crate::pipeline::AdapterPaths::Lora(lora_adapter_paths) = $paths.get_adapter_paths()
        else {
            unreachable!()
        };

        let regexes = if $loading_isq && $loading_uqff {
            // Dummy weights for the layers which will be overwritten...
            Some(std::sync::Arc::new(if $is_moqe {
                $loader.isq_layer_regexes_moqe(&$config)?
            } else {
                $loader.isq_layer_regexes(&$config)?
            }))
        } else {
            None
        };
        let get_device_for_tensor =
            $loader.get_device_for_tensor(&$config, &*$mapper, $loading_isq)?;

        let vb = from_mmaped_safetensors(
            $paths.get_weight_filenames().to_vec(),
            Vec::new(),
            $dtype,
            $device,
            $layer_devices,
            $silent,
            regexes,
            |_| true, // Will be overwritten...
            get_device_for_tensor.clone(),
        )?;

        for $crate::pipeline::LoraAdapterPaths {
            adapter_path,
            lora_config,
        } in lora_adapter_paths
        {
            let lora_vb = from_mmaped_safetensors(
                vec![adapter_path.clone()],
                Vec::new(),
                $dtype,
                $device,
                $layer_devices,
                $silent,
                None,
                |_| true,
                get_device_for_tensor.clone(),
            )?;

            mistralrs_quant::push_applied_lora(mistralrs_quant::LoraAdapter {
                config: lora_config.clone(),
                weights: lora_vb,
            });
        }

        $loader.load(
            &$config,
            vb,
            $crate::pipeline::NormalLoadingMetadata {
                mapper: $mapper,
                loading_isq: $loading_isq,
                real_device: $real_device,
                multi_progress: $multi_progress,
                matformer_slicing_config: $matformer_config,
            },
            $attention_mechanism,
        )?
    }};
}
