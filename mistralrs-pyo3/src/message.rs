use pyo3::{pyclass, pymethods};

#[pyclass]
#[derive(Clone, Debug)]
pub enum Role {
    User,
    Assistant,
    System,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[pymethods]
impl Message {
    #[new]
    #[pyo3(signature = (role, content))]
    fn new(role: Role, content: String) -> Self {
        Self { role, content }
    }
}
