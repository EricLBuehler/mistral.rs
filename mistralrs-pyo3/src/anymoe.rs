use pyo3::{pyclass, pymethods};

#[pyclass]
#[derive(Clone, Debug)]
pub enum AnyMoeExpertType {
    FineTuned {},
    LoraAdapter {
        rank: usize,
        alpha: f64,
        target_modules: Vec<String>,
    },
}

impl From<AnyMoeExpertType> for mistralrs_core::AnyMoeExpertType {
    fn from(val: AnyMoeExpertType) -> Self {
        match val {
            AnyMoeExpertType::FineTuned {} => Self::FineTuned,
            AnyMoeExpertType::LoraAdapter {
                rank,
                alpha,
                target_modules,
            } => Self::LoraAdapter {
                rank,
                alpha,
                target_modules,
            },
        }
    }
}

#[derive(Clone)]
#[pyclass]
pub struct AnyMoeConfig {
    pub(crate) hidden_size: usize,
    pub(crate) lr: f64,
    pub(crate) epochs: usize,
    pub(crate) batch_size: usize,
    pub(crate) expert_type: AnyMoeExpertType,
    pub(crate) dataset_csv: String,
    pub(crate) prefix: String,
    pub(crate) mlp: String,
    pub(crate) model_ids: Vec<String>,
    pub(crate) layers: Vec<usize>,
    pub(crate) gate_model_id: Option<String>,
    pub(crate) training: bool,
}

#[pymethods]
impl AnyMoeConfig {
    #[new]
    #[pyo3(signature = (
        hidden_size,
        dataset_csv,
        prefix,
        mlp,
        model_ids,
        expert_type,
        layers = vec![],
        lr = 1e-3,
        epochs = 100,
        batch_size = 4,
        gate_model_id = None,
        training = true,
    ))]
    fn new(
        hidden_size: usize,
        dataset_csv: String,
        prefix: String,
        mlp: String,
        model_ids: Vec<String>,
        expert_type: AnyMoeExpertType,
        layers: Vec<usize>,
        lr: f64,
        epochs: usize,
        batch_size: usize,
        gate_model_id: Option<String>,
        training: bool,
    ) -> Self {
        Self {
            hidden_size,
            lr,
            epochs,
            batch_size,
            expert_type,
            dataset_csv,
            prefix,
            mlp,
            model_ids,
            layers,
            gate_model_id,
            training,
        }
    }
}
