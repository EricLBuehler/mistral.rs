#[doc(hidden)]
#[macro_export]
macro_rules! api_dir_list {
    ($api:expr, $model_id:expr) => {
        if std::path::Path::new($model_id).exists() {
            let listing = std::fs::read_dir($model_id);
            if listing.is_err() {
                panic!("Cannot list directory {:?}", $model_id)
            }
            let listing = listing.unwrap();
            listing
                .into_iter()
                .map(|s| {
                    s.unwrap()
                        .path()
                        .file_name()
                        .unwrap() // Should never terminate in `..`
                        .to_str()
                        .expect("Could not convert to str")
                        .to_string()
                })
                .collect::<Vec<String>>()
                .into_iter()
        } else {
            $api.info()
                .map(|repo| {
                    repo.siblings
                        .iter()
                        .map(|x| x.rfilename.clone())
                        .collect::<Vec<String>>()
                })
                .unwrap_or_else(|e| panic!("Could not get directory listing from API: {:?}", e))
                .into_iter()
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! api_get_file {
    ($api:expr, $file:expr, $model_id:expr) => {
        if std::path::Path::new($model_id).exists() {
            let path = $model_id.join($file);
            if !path.exists() {
                panic!("File \"{}\" not found at model id {:?}", $file, $model_id)
            }
            info!("Loading `{}` locally at `{}`", &$file, path.display());
            path
        } else {
            $api.get($file)
                .unwrap_or_else(|e| panic!("Could not get file {:?} from API: {:?}", $file, e))
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! get_paths {
    ($path_name:ident, $token_source:expr, $revision:expr, $this:expr, $quantized_model_id:expr, $quantized_filename:expr, $silent:expr) => {{
        let api = ApiBuilder::new()
            .with_progress(!$silent)
            .with_token(get_token($token_source)?)
            .build()?;
        let revision = $revision.unwrap_or("main".to_string());
        let api = api.repo(Repo::with_revision(
            $this.model_id.clone(),
            RepoType::Model,
            revision.clone(),
        ));
        let model_id = std::path::Path::new(&$this.model_id);
        let tokenizer_filename = if let Some(ref p) = $this.tokenizer_json {
            info!("Using tokenizer.json at `{p}`");
            PathBuf::from_str(p)?
        } else {
            info!("Loading `tokenizer.json` at `{}`", $this.model_id);
            $crate::api_get_file!(api, "tokenizer.json", model_id)
        };
        info!("Loading `config.json` at `{}`", $this.model_id);
        let config_filename = $crate::api_get_file!(api, "config.json", model_id);
        let filenames = get_model_paths(
            revision.clone(),
            &$token_source,
            &$quantized_model_id,
            &$quantized_filename,
            &api,
            &model_id,
        )?;
        let XLoraPaths {
            adapter_configs,
            adapter_safetensors,
            classifier_path,
            xlora_order,
            xlora_config,
            lora_preload_adapter_info,
        } = get_xlora_paths(
            $this.model_id.clone(),
            &$this.xlora_model_id,
            &$token_source,
            revision.clone(),
            &$this.xlora_order,
        )?;
        let gen_conf = if $crate::api_dir_list!(api, model_id)
            .collect::<Vec<_>>()
            .contains(&"generation_config.json".to_string())
        {
            info!("Loading `generation_config.json` at `{}`", $this.model_id);
            Some($crate::api_get_file!(
                api,
                "generation_config.json",
                model_id
            ))
        } else {
            None
        };
        let preprocessor_config = if $crate::api_dir_list!(api, model_id)
            .collect::<Vec<_>>()
            .contains(&"preprocessor_config.json".to_string())
        {
            info!("Loading `preprocessor_config.json` at `{}`", $this.model_id);
            Some($crate::api_get_file!(
                api,
                "preprocessor_config.json",
                model_id
            ))
        } else {
            None
        };
        let processor_config = if $crate::api_dir_list!(api, model_id)
            .collect::<Vec<_>>()
            .contains(&"processor_config.json".to_string())
        {
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
        } else {
            info!("Loading `tokenizer_config.json` at `{}`", $this.model_id);
            Some($crate::api_get_file!(
                api,
                "tokenizer_config.json",
                model_id
            ))
        };
        Ok(Box::new($path_name {
            tokenizer_filename,
            config_filename,
            filenames,
            xlora_adapter_configs: adapter_configs,
            xlora_adapter_filenames: adapter_safetensors,
            classifier_path,
            classifier_config: xlora_config,
            xlora_ordering: xlora_order,
            template_filename,
            gen_conf,
            lora_preload_adapter_info,
            preprocessor_config,
            processor_config,
        }))
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! get_write_uqff_paths {
    ($from_uqff:expr, $this:expr, $silent:expr) => {{
        if !$from_uqff.exists() {
            // Assume it's a HF model id
            let path = $from_uqff.to_string_lossy().to_string();
            let parts = path.rsplitn(2, '/').collect::<Vec<_>>();

            if parts.len() != 2 {
                anyhow::bail!("ISQ artifact load path `{path}` not found locally must have format `<HF MODEL ID>/<FILENAME>`");
            }

            let file = parts[0];
            let model_id = parts[1];

            let api = ApiBuilder::new()
                .with_progress(!$silent)
                .with_token(get_token(
                    &$this
                        .token_source
                        .read()
                        .expect("Failed to read token source")
                        .clone()
                        .unwrap_or(TokenSource::None),
                )?)
                .build()?;
            let revision = $this
                .revision
                .read()
                .expect("Failed to read revision")
                .clone()
                .unwrap_or("main".to_string());
            let api = api.repo(Repo::with_revision(
                model_id.to_string(),
                RepoType::Model,
                revision.clone(),
            ));

            api_get_file!(api, file, Path::new(model_id))
        } else {
            $from_uqff
        }
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! get_paths_gguf {
    ($path_name:ident, $token_source:expr, $revision:expr, $this:expr, $quantized_model_id:expr, $quantized_filenames:expr, $silent:expr) => {{
        let api = ApiBuilder::new()
            .with_progress(!$silent)
            .with_token(get_token($token_source)?)
            .build()?;
        let revision = $revision.unwrap_or("main".to_string());
        let this_model_id = $this.model_id.clone().unwrap_or($this.quantized_model_id.clone());
        let api = api.repo(Repo::with_revision(
            this_model_id.clone(),
            RepoType::Model,
            revision.clone(),
        ));
        let model_id = std::path::Path::new(&this_model_id);

        let chat_template = if let Some(ref p) = $this.chat_template {
            if p.ends_with(".json") {
                info!("Using chat template file at `{p}`");
                Some(PathBuf::from_str(p)?)
            } else {
                panic!("Specified chat template file must end with .json");
            }
        } else {
            if $this.model_id.is_none() {
                None
            } else {
                info!("Loading `tokenizer_config.json` at `{}` because no chat template file was specified.", this_model_id);
                let res = $crate::api_get_file!(
                    api,
                    "tokenizer_config.json",
                    model_id
                );
                Some(res)
            }
        };

        let filenames = get_model_paths(
            revision.clone(),
            &$token_source,
            &Some($quantized_model_id),
            &Some($quantized_filenames),
            &api,
            &model_id,
        )?;

        let XLoraPaths {
            adapter_configs,
            adapter_safetensors,
            classifier_path,
            xlora_order,
            xlora_config,
            lora_preload_adapter_info,
        } = get_xlora_paths(
            this_model_id.clone(),
            &$this.xlora_model_id,
            &$token_source,
            revision.clone(),
            &$this.xlora_order,
        )?;

        let gen_conf = if $crate::api_dir_list!(api, model_id)
            .collect::<Vec<_>>()
            .contains(&"generation_config.json".to_string())
        {
            info!("Loading `generation_config.json` at `{}`", this_model_id);
            Some($crate::api_get_file!(
                api,
                "generation_config.json",
                model_id
            ))
        } else {
            None
        };

        let preprocessor_config = if $crate::api_dir_list!(api, model_id)
            .collect::<Vec<_>>()
            .contains(&"preprocessor_config.json".to_string())
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

        let processor_config = if $crate::api_dir_list!(api, model_id)
            .collect::<Vec<_>>()
            .contains(&"processor_config.json".to_string())
        {
            info!("Loading `processor_config.json` at `{}`", this_model_id);
            Some($crate::api_get_file!(
                api,
                "processor_config.json",
                model_id
            ))
        } else {
            None
        };

        let tokenizer_filename = if $this.model_id.is_some() {
            info!("Loading `tokenizer.json` at `{}`", this_model_id);
            $crate::api_get_file!(api, "tokenizer.json", model_id)
        } else {
            PathBuf::from_str("")?
        };

        Ok(Box::new($path_name {
            tokenizer_filename,
            config_filename: PathBuf::from_str("")?,
            filenames,
            xlora_adapter_configs: adapter_configs,
            xlora_adapter_filenames: adapter_safetensors,
            classifier_path,
            classifier_config: xlora_config,
            xlora_ordering: xlora_order,
            template_filename: chat_template,
            gen_conf,
            lora_preload_adapter_info,
            preprocessor_config,
            processor_config
        }))
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! normal_model_loader {
    ($paths:expr, $dtype:expr, $device:expr, $config:expr, $loader:expr, $use_flash_attn:expr, $silent:expr, $mapper:expr, $loading_isq:expr, $loading_uqff:expr, $real_device:expr, $attention_mechanism:expr, $is_moqe:expr, $cache_type:expr) => {{
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

        let vb = from_mmaped_safetensors(
            $paths.get_weight_filenames().to_vec(),
            Vec::new(),
            $dtype,
            $device,
            $silent,
            regexes,
            |_| true, // Will be overwritten...
        )?;

        $loader.load(
            &$config,
            $use_flash_attn,
            vb,
            $crate::pipeline::NormalLoadingMetadata {
                mapper: $mapper,
                loading_isq: $loading_isq,
                real_device: $real_device,
                cache_type: $cache_type,
            },
            $attention_mechanism,
        )?
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! vision_normal_model_loader {
    ($paths:expr, $dtype:expr, $device:expr, $config:expr, $loader:expr, $use_flash_attn:expr, $silent:expr, $mapper:expr, $loading_isq:expr, $loading_uqff:expr, $real_device:expr, $attention_mechanism:expr, $cache_type:expr) => {{
        let regexes = if $loading_isq && $loading_uqff {
            // Dummy weights for the layers which will be overwritten...
            Some(std::sync::Arc::new($loader.isq_layer_regexes(&$config)?))
        } else {
            None
        };

        let vb = from_mmaped_safetensors(
            $paths.get_weight_filenames().to_vec(),
            Vec::new(),
            $dtype,
            $device,
            $silent,
            regexes,
            |_| true,
        )?;

        $loader.load(
            &$config,
            $use_flash_attn,
            vb,
            $crate::pipeline::NormalLoadingMetadata {
                mapper: $mapper,
                loading_isq: $loading_isq,
                real_device: $real_device,
                cache_type: $cache_type,
            },
            $attention_mechanism,
        )?
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! xlora_model_loader {
    ($paths:expr, $dtype:expr, $device:expr, $config:expr, $loader:expr, $use_flash_attn:expr, $silent:expr, $mapper:expr, $loading_isq:expr, $real_device:expr) => {{
        let mut safetensors_paths = $paths.get_weight_filenames().iter().collect::<Vec<_>>();
        safetensors_paths.push($paths.get_classifier_path().as_ref().unwrap());
        let vb = from_mmaped_safetensors(
            safetensors_paths
                .iter()
                .map(|x| (*x).to_owned())
                .collect::<Vec<_>>(),
            $paths
                .get_adapter_filenames()
                .as_ref()
                .unwrap()
                .iter()
                .map(|(_, x)| (*x).to_owned())
                .collect::<Vec<_>>(),
            $dtype,
            $device,
            $silent,
            None,
            |_| true,
        )?;

        $loader.load_xlora(
            &$config,
            $use_flash_attn,
            vb,
            $paths.get_adapter_configs().as_ref().unwrap(),
            Some($paths.get_classifier_config().as_ref().unwrap().clone()),
            $paths.get_ordering().as_ref().unwrap().clone(),
            $crate::pipeline::NormalLoadingMetadata {
                mapper: $mapper,
                loading_isq: $loading_isq,
                real_device: $real_device,
                cache_type: None,
            },
            &None,
        )?
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! lora_model_loader {
    ($paths:expr, $dtype:expr, $device:expr, $config:expr, $loader:expr, $use_flash_attn:expr, $silent:expr, $mapper:expr, $loading_isq:expr, $real_device:expr) => {{
        let safetensors_paths = $paths.get_weight_filenames().iter().collect::<Vec<_>>();
        let vb = from_mmaped_safetensors(
            safetensors_paths
                .iter()
                .map(|x| (*x).to_owned())
                .collect::<Vec<_>>(),
            $paths
                .get_adapter_filenames()
                .as_ref()
                .unwrap()
                .iter()
                .map(|(_, x)| (*x).to_owned())
                .collect::<Vec<_>>(),
            Some($dtype),
            $device,
            $silent,
            None,
            |_| true,
        )?;

        $loader.load_xlora(
            &$config,
            $use_flash_attn,
            vb,
            $paths.get_adapter_configs().as_ref().unwrap(),
            None,
            $paths.get_ordering().as_ref().unwrap().clone(),
            $crate::pipeline::NormalLoadingMetadata {
                mapper: $mapper,
                loading_isq: $loading_isq,
                real_device: $real_device,
                cache_type: None,
            },
            &$crate::utils::varbuilder_utils::load_preload_adapters(
                $paths.get_lora_preload_adapter_info(),
                $dtype,
                $device,
                $silent,
            )?,
        )?
    }};
}
