#[macro_export]
macro_rules! api_dir_list {
    ($api:expr, $model_id:expr) => {
        $api.info()
            .map(|repo| {
                repo.siblings
                    .iter()
                    .map(|x| x.rfilename.clone())
                    .collect::<Vec<String>>()
            })
            .unwrap_or_else(|e| {
                // If we do not get a 404, it was something else.
                let format = format!("{e:?}");
                if let hf_hub::api::sync::ApiError::RequestError(resp) = e {
                    if resp.into_response().is_some_and(|r| r.status() != 404) {
                        panic!("{format}");
                    }
                }

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
                            .to_str()
                            .expect("Could not convert to str")
                            .to_string()
                    })
                    .collect::<Vec<String>>()
            })
            .into_iter()
    };
}

#[macro_export]
macro_rules! api_get_file {
    ($api:expr, $file:expr, $model_id:expr) => {
        $api.get($file).unwrap_or_else(|e| {
            // If we do not get a 404, it was something else.
            let format = format!("{e:?}");
            if let hf_hub::api::sync::ApiError::RequestError(resp) = e {
                if resp.into_response().is_some_and(|r| r.status() != 404) {
                    panic!("{format}");
                }
            }

            let path = $model_id.join($file);
            if !path.exists() {
                panic!("File \"{}\" not found at model id {:?}", $file, $model_id)
            }
            info!("Loading `{:?}` locally at `{path:?}`", &$file);
            path
        })
    };
}

#[macro_export]
macro_rules! deserialize_chat_template {
    ($paths:expr, $this:ident) => {{
        use tracing::info;

        let template: ChatTemplate = serde_json::from_str(&fs::read_to_string(
            $paths.get_template_filename(),
        )?).unwrap();
        let gen_conf: Option<$crate::pipeline::chat_template::GenerationConfig> = $paths.get_gen_conf_filename()
            .map(|f| serde_json::from_str(&fs::read_to_string(
                f
            ).unwrap()).unwrap());
        #[derive(Debug, serde::Deserialize)]
        struct SpecifiedTemplate {
            chat_template: String,
            bos_token: Option<String>,
            eos_token: Option<String>,
        }
        match template.chat_template {
            Some(_) => (template, gen_conf),
            None => {
                info!("`tokenizer_config.json` does not contain a chat template, attempting to use specified JINJA chat template.");
                let mut deser: HashMap<String, Value> =
                    serde_json::from_str(&fs::read_to_string($paths.get_template_filename())?)
                        .unwrap();
                match $this.chat_template.clone() {
                    Some(t) => {
                        if t.ends_with(".json") {
                            info!("Loading specified loading chat template file at `{t}`.");
                            let templ: SpecifiedTemplate = serde_json::from_str(&fs::read_to_string(t.clone())?).unwrap();
                            deser.insert(
                                "chat_template".to_string(),
                                Value::String(templ.chat_template),
                            );
                            if templ.bos_token.is_some() {
                                deser.insert(
                                    "bos_token".to_string(),
                                    Value::String(templ.bos_token.unwrap()),
                                );
                            }
                            if templ.eos_token.is_some() {
                                deser.insert(
                                    "eos_token".to_string(),
                                    Value::String(templ.eos_token.unwrap()),
                                );
                            }
                            info!("Loaded chat template file.");
                        } else {
                            deser.insert(
                                "chat_template".to_string(),
                                Value::String(t),
                            );
                            info!("Loaded specified literal chat template.");
                        }
                    },
                    None => {
                        info!("No specified chat template. No chat template will be used. Only prompts will be accepted, not messages.");
                        deser.insert(
                            "chat_template".to_string(),
                            Value::Null,
                        );
                    }
                };
                let ser = serde_json::to_string_pretty(&deser).expect("Serialization of modified chat template failed.");
                (serde_json::from_str(&ser).unwrap(), gen_conf)
            }
        }
    }};
}

#[macro_export]
macro_rules! get_paths {
    ($path_name:ident, $token_source:expr, $revision:expr, $this:expr, $quantized_model_id:expr, $quantized_filename:expr, $silent:expr) => {{
        let api = ApiBuilder::new()
            .with_progress(!$silent)
            .with_token(Some(get_token($token_source)?))
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
            $crate::api_get_file!(api, "tokenizer.json", model_id)
        };

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
            Some($crate::api_get_file!(
                api,
                "generation_config.json",
                model_id
            ))
        } else {
            None
        };

        let template_filename = $crate::api_get_file!(api, "tokenizer_config.json", model_id);

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
        }))
    }};
}

#[macro_export]
macro_rules! normal_model_loader {
    ($paths:expr, $dtype:expr, $default_dtype:expr, $device:expr, $config:expr, $loader:expr, $use_flash_attn:expr, $silent:expr, $mapper:expr, $loading_isq:expr, $real_device:expr) => {{
        let vb = from_mmaped_safetensors(
            $paths.get_weight_filenames().to_vec(),
            Vec::new(),
            $dtype.unwrap_or($default_dtype),
            $device,
            $silent,
        )?;

        $loader.load(
            &$config,
            $use_flash_attn,
            vb,
            $mapper,
            $loading_isq,
            $real_device,
        )?
    }};
}

#[macro_export]
macro_rules! xlora_model_loader {
    ($paths:expr, $dtype:expr, $default_dtype:expr, $device:expr, $config:expr, $loader:expr, $use_flash_attn:expr, $silent:expr, $mapper:expr, $loading_isq:expr, $real_device:expr) => {{
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
            $dtype.unwrap_or($default_dtype),
            $device,
            $silent,
        )?;

        $loader.load_xlora(
            &$config,
            $use_flash_attn,
            vb,
            $paths.get_adapter_configs().as_ref().unwrap(),
            Some($paths.get_classifier_config().as_ref().unwrap().clone()),
            $paths.get_ordering().as_ref().unwrap().clone(),
            $mapper,
            $loading_isq,
            $real_device,
            &$crate::utils::varbuilder_utils::load_preload_adapters(
                $paths.get_lora_preload_adapter_info(),
                $dtype.unwrap_or($default_dtype),
                $device,
                $silent,
            )?,
        )?
    }};
}

#[macro_export]
macro_rules! lora_model_loader {
    ($paths:expr, $dtype:expr, $default_dtype:expr, $device:expr, $config:expr, $loader:expr, $use_flash_attn:expr, $silent:expr, $mapper:expr, $loading_isq:expr, $real_device:expr) => {{
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
            $dtype.unwrap_or($default_dtype),
            $device,
            $silent,
        )?;

        $loader.load_xlora(
            &$config,
            $use_flash_attn,
            vb,
            $paths.get_adapter_configs().as_ref().unwrap(),
            None,
            $paths.get_ordering().as_ref().unwrap().clone(),
            $mapper,
            $loading_isq,
            $real_device,
            &$crate::utils::varbuilder_utils::load_preload_adapters(
                $paths.get_lora_preload_adapter_info(),
                $dtype.unwrap_or($default_dtype),
                $device,
                $silent,
            )?,
        )?
    }};
}
