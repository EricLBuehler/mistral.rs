use mistralrs::{IsqType, VisionLoaderType};

use crate::SelectedModel;

pub enum ModelSpec {
    Text {
        uqff_model_id: &'static str,
        supported_quants: &'static [IsqType],
        stem: &'static str,
    },
    Vision {
        uqff_model_id: &'static str,
        supported_quants: &'static [IsqType],
        arch: VisionLoaderType,
        stem: &'static str,
    },
}

impl ModelSpec {
    pub fn from_selected(selected: SelectedModel) -> Self {
        match selected {
            SelectedModel::VLlama_11b => ModelSpec::Vision {
                uqff_model_id: "EricB/Llama-3.2-11B-Vision-Instruct-UQFF",
                supported_quants: &[
                    IsqType::F8E4M3,
                    IsqType::Q3K,
                    IsqType::Q4K,
                    IsqType::Q5K,
                    IsqType::Q8_0,
                    IsqType::HQQ4,
                    IsqType::HQQ4,
                ],
                arch: VisionLoaderType::VLlama,
                stem: "llama3.2-vision-instruct",
            },
            SelectedModel::MistralNemo_12b => ModelSpec::Text {
                uqff_model_id: "EricB/Mistral-Nemo-Instruct-2407-UQFF",
                supported_quants: &[
                    IsqType::F8E4M3,
                    IsqType::Q3K,
                    IsqType::Q4K,
                    IsqType::Q5K,
                    IsqType::Q8_0,
                    IsqType::HQQ4,
                    IsqType::HQQ4,
                ],
                stem: "mistral-nemo-2407-instruct",
            },
            SelectedModel::Phi3_3_8b => ModelSpec::Text {
                uqff_model_id: "EricB/Phi-3.5-mini-instruct-UQFF",
                supported_quants: &[
                    IsqType::F8E4M3,
                    IsqType::Q3K,
                    IsqType::Q4K,
                    IsqType::Q5K,
                    IsqType::Q8_0,
                    IsqType::HQQ4,
                    IsqType::HQQ4,
                ],
                stem: "phi3.5-mini-instruct",
            },
            SelectedModel::Gemma1_2b => ModelSpec::Text {
                uqff_model_id: "EricB/gemma-1.1-2b-it-UQFF",
                supported_quants: &[
                    IsqType::F8E4M3,
                    IsqType::Q3K,
                    IsqType::Q4K,
                    IsqType::Q5K,
                    IsqType::Q8_0,
                    IsqType::HQQ4,
                    IsqType::HQQ4,
                ],
                stem: "gemma1.1-2b-instruct",
            },
            SelectedModel::Gemma1_7b => ModelSpec::Text {
                uqff_model_id: "EricB/gemma-1.1-7b-it-UQFF",
                supported_quants: &[
                    IsqType::F8E4M3,
                    IsqType::Q3K,
                    IsqType::Q4K,
                    IsqType::Q5K,
                    IsqType::Q8_0,
                    IsqType::HQQ4,
                    IsqType::HQQ4,
                ],
                stem: "gemma1.1-7b-instruct",
            },
            SelectedModel::Gemma2_2b => ModelSpec::Text {
                uqff_model_id: "EricB/gemma-2-2b-it-UQFF",
                supported_quants: &[
                    IsqType::F8E4M3,
                    IsqType::Q3K,
                    IsqType::Q4K,
                    IsqType::Q5K,
                    IsqType::Q8_0,
                    IsqType::HQQ4,
                    IsqType::HQQ4,
                ],
                stem: "gemma2-2b-instruct",
            },
            SelectedModel::Gemma2_9b => ModelSpec::Text {
                uqff_model_id: "EricB/gemma-2-9b-it-UQFF",
                supported_quants: &[
                    IsqType::F8E4M3,
                    IsqType::Q3K,
                    IsqType::Q4K,
                    IsqType::Q5K,
                    IsqType::Q8_0,
                    IsqType::HQQ4,
                    IsqType::HQQ4,
                ],
                stem: "gemma2-29b-instruct",
            },
            SelectedModel::Gemma2_27b => ModelSpec::Text {
                uqff_model_id: "EricB/gemma-2-27b-it-UQFF",
                supported_quants: &[
                    IsqType::F8E4M3,
                    IsqType::Q3K,
                    IsqType::Q4K,
                    IsqType::Q5K,
                    IsqType::Q8_0,
                    IsqType::HQQ4,
                    IsqType::HQQ4,
                ],
                stem: "gemma2-27b-instruct",
            },
            SelectedModel::Llama3_2_3b => ModelSpec::Text {
                uqff_model_id: "EricB/Llama-3.2-3B-Instruct-UQFF",
                supported_quants: &[
                    IsqType::F8E4M3,
                    IsqType::Q3K,
                    IsqType::Q4K,
                    IsqType::Q5K,
                    IsqType::Q8_0,
                    IsqType::HQQ4,
                    IsqType::HQQ4,
                ],
                stem: "llama3.2-3b-instruct",
            },
            SelectedModel::Llama3_2_1b => ModelSpec::Text {
                uqff_model_id: "EricB/Llama-3.2-1B-Instruct-UQFF",
                supported_quants: &[
                    IsqType::F8E4M3,
                    IsqType::Q3K,
                    IsqType::Q4K,
                    IsqType::Q5K,
                    IsqType::Q8_0,
                    IsqType::HQQ4,
                    IsqType::HQQ4,
                ],
                stem: "llama3.2-1b-instruct",
            },
            SelectedModel::Llama3_1_8b => ModelSpec::Text {
                uqff_model_id: "EricB/Llama-3.1-8B-Instruct-UQFF",
                supported_quants: &[
                    IsqType::F8E4M3,
                    IsqType::Q3K,
                    IsqType::Q4K,
                    IsqType::Q5K,
                    IsqType::Q8_0,
                    IsqType::HQQ4,
                    IsqType::HQQ4,
                ],
                stem: "llama3.1-8b-instruct",
            },
            SelectedModel::Mistral_7b => ModelSpec::Text {
                uqff_model_id: "EricB/Mistral-7B-Instruct-v0.3-UQFF",
                supported_quants: &[
                    IsqType::F8E4M3,
                    IsqType::Q3K,
                    IsqType::Q4K,
                    IsqType::Q5K,
                    IsqType::Q8_0,
                    IsqType::HQQ4,
                    IsqType::HQQ4,
                ],
                stem: "mistral0.3-7b-instruct",
            },
            SelectedModel::MistralSmall_12b => ModelSpec::Text {
                uqff_model_id: "EricB/Mistral-Small-Instruct-2409-UQFF",
                supported_quants: &[
                    IsqType::F8E4M3,
                    IsqType::Q3K,
                    IsqType::Q4K,
                    IsqType::Q5K,
                    IsqType::Q8_0,
                    IsqType::HQQ4,
                    IsqType::HQQ4,
                ],
                stem: "mistral-small-2409-instruct",
            },
            SelectedModel::Phi3V_3_8b => ModelSpec::Vision {
                uqff_model_id: "EricB/Phi-3.5-vision-instruct-UQFF",
                supported_quants: &[
                    IsqType::F8E4M3,
                    IsqType::Q3K,
                    IsqType::Q4K,
                    IsqType::Q5K,
                    IsqType::Q8_0,
                    IsqType::HQQ4,
                    IsqType::HQQ4,
                ],
                arch: VisionLoaderType::Phi3V,
                stem: "phi3.5-vision-instruct",
            },
        }
    }

    pub fn default_quant(&self) -> IsqType {
        IsqType::Q4K
    }
}
