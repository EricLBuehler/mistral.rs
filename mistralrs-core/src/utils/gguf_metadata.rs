use akin::akin;
use anyhow::ensure;
use anyhow::Result;
use candle_core::quantized::gguf_file;
use std::collections::HashMap;

pub struct ContentMetadata<'a> {
    pub path_prefix: String,
    pub metadata: &'a HashMap<String, gguf_file::Value>,
}

impl ContentMetadata<'_> {
    // Retrieve a prop the struct needs by querying the metadata content:
    pub fn get_value<T: TryFromValue>(&self, field_name: &str) -> Result<T, candle_core::Error> {
        let prop_key = format!("{prefix}.{field_name}", prefix = self.path_prefix);
        let value = self.metadata.get(&prop_key).cloned();

        // Unwrap the inner value of the `Value` enum via trait method,
        // otherwise format error with prop key as context:
        value
            .try_value_into()
            .or_else(|e| candle_core::bail!("`{prop_key}` `{e}`"))
    }

    // Fail early - Catch all missing mandatory keys upfront:
    pub fn has_required_keys(&self, fields: &[&str]) -> Result<()> {
        let mut all_props_are_present = true;

        for field_name in fields {
            let prop_key = format!("{prefix}.{field_name}", prefix = self.path_prefix);

            if !self.metadata.contains_key(&prop_key) {
                all_props_are_present = false;
                eprintln!("Expected GGUF metadata to have key: `{prop_key}`");
            }
        }

        ensure!(all_props_are_present, "Tokenizer is missing required props");
        Ok(())
    }
}

pub trait TryFromValue {
    fn try_from_value(value: gguf_file::Value) -> Result<Self, candle_core::Error>
    where
        Self: Sized;
}

// Value wrapped types, each has a different conversion method:
// NOTE: Type conversion methods internally bail with "not a <into type> <input value>"
// https://docs.rs/candle-core/latest/candle_core/quantized/gguf_file/enum.Value.html#variants
akin! {
    let &types = [String, f32, u32];
    let &to_type = [value.to_string().cloned(), value.to_f32(), value.to_u32()];

    impl TryFromValue for *types {
        fn try_from_value(value: gguf_file::Value) -> Result<Self, candle_core::Error> {
            *to_type.or_else(|_| candle_core::bail!("value is not a `*types`"))
        }
    }
}

// Vec<Value> to Vec<T> from above types:
impl<T: TryFromValue> TryFromValue for Vec<T> {
    fn try_from_value(value_vec: gguf_file::Value) -> Result<Self, candle_core::Error> {
        value_vec
            .to_vec()
            .or_else(|_| candle_core::bail!("value is not a `Vec`"))?
            .clone()
            .into_iter()
            .map(|item| T::try_from_value(item))
            .collect()
    }
}

pub trait TryValueInto<T>: Sized {
    fn try_value_into(self) -> Result<T, candle_core::Error>;
}

impl<T: TryFromValue> TryValueInto<T> for gguf_file::Value {
    fn try_value_into(self) -> Result<T, candle_core::Error> {
        T::try_from_value(self)
    }
}

impl<T: TryFromValue> TryValueInto<T> for Option<gguf_file::Value> {
    fn try_value_into(self) -> Result<T, candle_core::Error> {
        match self {
            Some(value) => value.try_value_into(),
            None => candle_core::bail!("Option is missing value"),
        }
    }
}
