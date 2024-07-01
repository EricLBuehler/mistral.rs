use akin::akin;
use anyhow::ensure;
use anyhow::Result;
use candle_core::quantized::gguf_file;
use std::collections::HashMap;
use tracing::warn;

pub struct ContentMetadata<'a> {
    pub path_prefix: &'a str,
    pub metadata: &'a HashMap<String, gguf_file::Value>,
}

impl ContentMetadata<'_> {
    // Retrieve a prop the struct needs by querying the metadata content:
    pub fn get_value<T: TryFromValue>(&self, field_name: &str) -> Result<T, anyhow::Error> {
        let prop_key = format!("{prefix}.{field_name}", prefix = self.path_prefix);
        let value = self.metadata.get(&prop_key).cloned();

        // Unwrap the inner value of the `Value` enum via trait method,
        // otherwise format error with prop key as context:
        value
            .try_value_into()
            .or_else(|e| anyhow::bail!("`{prop_key}` `{e}`"))
    }

    // Retrieve a prop the struct needs by querying the metadata content:
    pub fn get_option_value<T: TryFromValue>(
        &self,
        field_name: &str,
    ) -> Result<Option<T>, anyhow::Error> {
        let prop_key = format!("{prefix}.{field_name}", prefix = self.path_prefix);
        let value = self.metadata.get(&prop_key).cloned();

        // Unwrap the inner value of the `Value` enum via trait method,
        // otherwise format error with prop key as context:
        value
            .map(|v| {
                v.try_value_into()
                    .or_else(|e| anyhow::bail!("`{prop_key}` `{e}`"))
            })
            .map_or(Ok(None), |res| res.map(Some))
    }

    // Fail early - Catch all missing mandatory keys upfront:
    pub fn has_required_keys(&self, fields: &[&str]) -> Result<()> {
        let mut all_props_are_present = true;

        for field_name in fields {
            let prop_key = format!("{prefix}.{field_name}", prefix = self.path_prefix);

            if !self.metadata.contains_key(&prop_key) {
                all_props_are_present = false;
                warn!("Expected GGUF metadata to have key: `{prop_key}`");
            }
        }

        ensure!(all_props_are_present, "Tokenizer is missing required props");
        Ok(())
    }

    // Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#required
    pub fn verify_arch(&self, expected_arch: &str) -> Result<()> {
        let actual_arch: String = self
            .metadata
            .get("general.architecture")
            .cloned()
            .try_value_into()?;

        anyhow::ensure!(
            actual_arch == expected_arch,
            "Expected `{expected_arch}` architecture, got `{actual_arch}`."
        );

        Ok(())
    }
}

// These traits below are a workaround for converting candles GGUF `Value` enum type wrapper.
// A better upstream approach would instead be to provide serialize/deserialize support?
pub trait TryFromValue {
    fn try_from_value(value: gguf_file::Value) -> Result<Self, candle_core::Error>
    where
        Self: Sized;
}

// Value wrapped types, each has a different conversion method:
// NOTE: Type conversion methods internally bail with "not a <into type> <input value>"
// https://docs.rs/candle-core/latest/candle_core/quantized/gguf_file/enum.Value.html#variants
akin! {
    let &types = [String, bool, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64];
    let &to_type = [
        value.to_string().cloned(),
        value.to_bool(),
        value.to_f32(),
        value.to_f64(),
        value.to_i8(),
        value.to_i16(),
        value.to_i32(),
        value.to_i64(),
        value.to_u8(),
        value.to_u16(),
        value.to_u32(),
        value.to_u64(),
    ];

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
            None => candle_core::bail!("Expected `Option<gguf_file::Value>` to contain a value"),
        }
    }
}
