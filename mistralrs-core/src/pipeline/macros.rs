#[macro_export]
macro_rules! deserialize_chat_template {
    ($paths:expr, $this:ident) => {{
        use tracing::info;

        let template: ChatTemplate = serde_json::from_str(&fs::read_to_string(
            $paths.get_template_filename(),
        )?).unwrap();
        #[derive(Debug, serde::Deserialize)]
        struct SpecifiedTemplate {
            chat_template: String,
            bos_token: Option<String>,
            eos_token: Option<String>,
        }
        match template.chat_template {
            Some(_) => template,
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
                serde_json::from_str(&ser).unwrap()
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

        let tokenizer_filename = if let Some(ref p) = $this.tokenizer_json {
            info!("Using tokenizer.json at `{p}`");
            PathBuf::from_str(p)?
        } else {
            api.get("tokenizer.json")?
        };

        let config_filename = api.get("config.json")?;

        let filenames = get_model_paths(
            revision.clone(),
            &$token_source,
            &$quantized_model_id,
            &$quantized_filename,
            &api,
        )?;

        let XLoraPaths {
            adapter_configs,
            adapter_safetensors,
            classifier_path,
            xlora_order,
            xlora_config,
        } = get_xlora_paths(
            $this.model_id.clone(),
            &$this.xlora_model_id,
            &$token_source,
            revision.clone(),
            &$this.xlora_order,
        )?;

        let template_filename = api.get("tokenizer_config.json")?;

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
        }))
    }};
}

#[macro_export]
macro_rules! normal_model_loader {
    ($paths:expr, $dtype:expr, $default_dtype:expr, $device:expr, $config:expr, $loader:expr, $use_flash_attn:expr, $silent:expr) => {{
        let vb = from_mmaped_safetensors(
            $paths.get_weight_filenames().to_vec(),
            Vec::new(),
            $dtype.unwrap_or($default_dtype),
            $device,
            $silent,
        )?;

        $loader.load(&$config, $use_flash_attn, vb)?
    }};
}

#[macro_export]
macro_rules! xlora_model_loader {
    ($paths:expr, $dtype:expr, $default_dtype:expr, $device:expr, $config:expr, $loader:expr, $use_flash_attn:expr, $silent:expr) => {{
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
        )?
    }};
}

#[macro_export]
macro_rules! lora_model_loader {
    ($paths:expr, $dtype:expr, $default_dtype:expr, $device:expr, $config:expr, $loader:expr, $use_flash_attn:expr, $silent:expr) => {{
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
        )?
    }};
}
