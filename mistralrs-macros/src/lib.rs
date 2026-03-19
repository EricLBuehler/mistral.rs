//! Proc macros for ergonomic tool definition in mistral.rs
//!
//! This crate provides the `#[tool]` attribute macro for defining tools
//! that can be used with the mistral.rs agentic loop.
//!
//! # Example
//!
//! ```ignore
//! use mistralrs_macros::tool;
//! use schemars::JsonSchema;
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Debug, Serialize, Deserialize, JsonSchema)]
//! struct WeatherInfo {
//!     temperature: f32,
//!     conditions: String,
//! }
//!
//! #[tool(description = "Get the current weather for a location")]
//! fn get_weather(
//!     #[description = "The city name"]
//!     city: String,
//! ) -> anyhow::Result<WeatherInfo> {
//!     Ok(WeatherInfo {
//!         temperature: 22.5,
//!         conditions: "Sunny".to_string(),
//!     })
//! }
//!
//! // This generates:
//! // - get_weather_tool() -> Tool
//! // - get_weather_callback() -> Arc<ToolCallback>
//! // - get_weather_tool_with_callback() -> (Tool, Arc<ToolCallback>)
//! ```

use darling::{ast::NestedMeta, FromMeta};
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Expr, FnArg, ItemFn, Lit, Meta, Pat, PatType, Type};

/// Arguments for the `#[tool]` attribute
#[derive(Debug, FromMeta)]
struct ToolArgs {
    /// Description of what the tool does
    description: String,
    /// Optional override for the tool name (defaults to function name)
    #[darling(default)]
    name: Option<String>,
}

/// Arguments for parameter-level attributes
#[derive(Debug, Default)]
struct ParamArgs {
    description: Option<String>,
    default: Option<Expr>,
}

impl ParamArgs {
    fn from_attrs(attrs: &[syn::Attribute]) -> Self {
        let mut args = ParamArgs::default();

        for attr in attrs {
            if attr.path().is_ident("description") {
                if let Meta::NameValue(nv) = &attr.meta {
                    if let Expr::Lit(expr_lit) = &nv.value {
                        if let Lit::Str(lit_str) = &expr_lit.lit {
                            args.description = Some(lit_str.value());
                        }
                    }
                }
            } else if attr.path().is_ident("default") {
                if let Meta::NameValue(nv) = &attr.meta {
                    args.default = Some(nv.value.clone());
                }
            }
        }

        args
    }
}

/// The `#[tool]` attribute macro for defining tools.
///
/// This macro transforms a regular Rust function into a tool that can be
/// used with the mistral.rs agentic loop. It generates:
///
/// - `{fn_name}_tool()` - Returns the `Tool` definition
/// - `{fn_name}_callback()` - Returns an `Arc<ToolCallback>` that wraps the function
/// - `{fn_name}_tool_with_callback()` - Returns both as a tuple
///
/// # Attributes
///
/// - `description` (required): A description of what the tool does
/// - `name` (optional): Override the tool name (defaults to function name)
///
/// # Parameter Attributes
///
/// - `#[description = "..."]`: Description of the parameter
/// - `#[default = value]`: Default value if parameter is optional
///
/// # Requirements
///
/// - All parameter types must implement `serde::Deserialize` and `schemars::JsonSchema`
/// - The return type must be `Result<T>` or `anyhow::Result<T>` where `T: Serialize`
/// - For async functions, ensure a tokio runtime is available
#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr_args = match NestedMeta::parse_meta_list(attr.into()) {
        Ok(v) => v,
        Err(e) => return TokenStream::from(e.into_compile_error()),
    };

    let tool_args = match ToolArgs::from_list(&attr_args) {
        Ok(v) => v,
        Err(e) => return TokenStream::from(e.write_errors()),
    };

    let input_fn = parse_macro_input!(item as ItemFn);

    match generate_tool_impl(tool_args, input_fn) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.into_compile_error().into(),
    }
}

fn generate_tool_impl(args: ToolArgs, input_fn: ItemFn) -> syn::Result<TokenStream2> {
    let fn_name = &input_fn.sig.ident;
    let fn_vis = &input_fn.vis;
    let is_async = input_fn.sig.asyncness.is_some();

    let tool_name = args.name.unwrap_or_else(|| fn_name.to_string());
    let description = &args.description;

    // Generate helper function names
    let tool_fn_name = format_ident!("{}_tool", fn_name);
    let callback_fn_name = format_ident!("{}_callback", fn_name);
    let combined_fn_name = format_ident!("{}_tool_with_callback", fn_name);
    let args_struct_name = format_ident!("__{}Args", fn_name);

    // Create a stripped version of the function without our custom attributes
    let mut stripped_fn = input_fn.clone();
    for arg in &mut stripped_fn.sig.inputs {
        if let FnArg::Typed(pat_type) = arg {
            // Remove #[description] and #[default] attributes
            pat_type.attrs.retain(|attr| {
                !attr.path().is_ident("description") && !attr.path().is_ident("default")
            });
        }
    }

    // Collect parameter information
    let mut param_names = Vec::new();
    let mut param_types = Vec::new();
    let mut param_descriptions = Vec::new();
    let mut param_defaults = Vec::new();
    let mut required_params = Vec::new();

    for arg in &input_fn.sig.inputs {
        if let FnArg::Typed(PatType { pat, ty, attrs, .. }) = arg {
            if let Pat::Ident(pat_ident) = pat.as_ref() {
                let param_name = &pat_ident.ident;
                let param_args = ParamArgs::from_attrs(attrs);

                param_names.push(param_name.clone());
                param_types.push(ty.as_ref().clone());
                param_descriptions.push(param_args.description);
                param_defaults.push(param_args.default);

                // Check if the type is Option<T>
                let is_optional = is_option_type(ty);
                if !is_optional && param_defaults.last().unwrap().is_none() {
                    required_params.push(param_name.to_string());
                }
            }
        }
    }

    // Generate the Args struct fields with serde attributes
    let args_struct_fields: Vec<TokenStream2> = param_names
        .iter()
        .zip(param_types.iter())
        .zip(param_defaults.iter())
        .map(|((name, ty), default)| {
            if default.is_some() {
                let default_fn_name_str = format!("__default_{}", name);
                quote! {
                    #[serde(default = #default_fn_name_str)]
                    pub #name: #ty
                }
            } else {
                quote! {
                    pub #name: #ty
                }
            }
        })
        .collect();

    // Generate default functions for parameters with defaults
    let default_fns: Vec<TokenStream2> = param_names
        .iter()
        .zip(param_types.iter())
        .zip(param_defaults.iter())
        .filter_map(|((name, ty), default)| {
            default.as_ref().map(|default_expr| {
                let default_fn_name = format_ident!("__default_{}", name);
                // If the type is Option<T>, wrap the default value in Some()
                let value_expr = if is_option_type(ty) {
                    quote! { Some(#default_expr.into()) }
                } else {
                    quote! { #default_expr }
                };
                quote! {
                    fn #default_fn_name() -> #ty {
                        #value_expr
                    }
                }
            })
        })
        .collect();

    // Generate property schema for each parameter
    let property_schemas: Vec<TokenStream2> = param_names
        .iter()
        .zip(param_types.iter())
        .zip(param_descriptions.iter())
        .map(|((name, ty), desc)| {
            let name_str = name.to_string();
            // Extract inner type if Option<T>
            let schema_type = extract_option_inner_type(ty).unwrap_or(ty);
            let desc_insert = if let Some(d) = desc {
                quote! {
                    if let Some(obj) = prop_schema.as_object_mut() {
                        obj.insert("description".to_string(), serde_json::json!(#d));
                    }
                }
            } else {
                quote! {}
            };
            quote! {
                {
                    let schema = schemars::schema_for!(#schema_type);
                    let mut prop_schema = serde_json::to_value(&schema).unwrap_or(serde_json::json!({}));
                    #desc_insert
                    properties.insert(#name_str.to_string(), prop_schema);
                }
            }
        })
        .collect();

    // Generate required array
    let required_array: Vec<TokenStream2> = required_params
        .iter()
        .map(|name| quote! { #name.to_string() })
        .collect();

    // Generate the function call
    let call_args: Vec<TokenStream2> = param_names
        .iter()
        .map(|name| quote! { args.#name })
        .collect();

    // Build the output based on whether function is async or sync
    let output = if is_async {
        // Async function: generate AsyncToolCallback
        quote! {
            // Original function preserved (with custom attributes stripped)
            #stripped_fn

            // Default value functions (if any)
            #(#default_fns)*

            // Arguments struct for deserialization
            #[derive(serde::Deserialize)]
            #[allow(non_camel_case_types)]
            struct #args_struct_name {
                #(#args_struct_fields),*
            }

            /// Returns the Tool definition for this function
            #fn_vis fn #tool_fn_name() -> mistralrs::Tool {
                let mut properties = std::collections::HashMap::<String, serde_json::Value>::new();

                #(#property_schemas)*

                let required: Vec<String> = vec![#(#required_array),*];

                let parameters: std::collections::HashMap<String, serde_json::Value> = serde_json::from_value(
                    serde_json::json!({
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    })
                ).expect("Failed to create tool parameters");

                mistralrs::Tool {
                    tp: mistralrs::ToolType::Function,
                    function: mistralrs::Function {
                        description: Some(#description.to_string()),
                        name: #tool_name.to_string(),
                        parameters: Some(parameters),
                    },
                }
            }

            /// Returns an async callback that wraps this function for tool execution
            #fn_vis fn #callback_fn_name() -> std::sync::Arc<mistralrs::AsyncToolCallback> {
                std::sync::Arc::new(|called: mistralrs::CalledFunction| {
                    Box::pin(async move {
                        let args: #args_struct_name = serde_json::from_str(&called.arguments)
                            .map_err(|e| anyhow::anyhow!("Failed to parse tool arguments: {}", e))?;

                        let result = #fn_name(#(#call_args),*).await?;

                        serde_json::to_string(&result)
                            .map_err(|e| anyhow::anyhow!("Failed to serialize tool result: {}", e))
                    })
                })
            }

            /// Returns both the Tool definition and callback as a tuple
            #fn_vis fn #combined_fn_name() -> (mistralrs::Tool, mistralrs::ToolCallbackType) {
                (#tool_fn_name(), mistralrs::ToolCallbackType::Async(#callback_fn_name()))
            }
        }
    } else {
        // Sync function: generate ToolCallback
        quote! {
            // Original function preserved (with custom attributes stripped)
            #stripped_fn

            // Default value functions (if any)
            #(#default_fns)*

            // Arguments struct for deserialization
            #[derive(serde::Deserialize)]
            #[allow(non_camel_case_types)]
            struct #args_struct_name {
                #(#args_struct_fields),*
            }

            /// Returns the Tool definition for this function
            #fn_vis fn #tool_fn_name() -> mistralrs::Tool {
                let mut properties = std::collections::HashMap::<String, serde_json::Value>::new();

                #(#property_schemas)*

                let required: Vec<String> = vec![#(#required_array),*];

                let parameters: std::collections::HashMap<String, serde_json::Value> = serde_json::from_value(
                    serde_json::json!({
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    })
                ).expect("Failed to create tool parameters");

                mistralrs::Tool {
                    tp: mistralrs::ToolType::Function,
                    function: mistralrs::Function {
                        description: Some(#description.to_string()),
                        name: #tool_name.to_string(),
                        parameters: Some(parameters),
                    },
                }
            }

            /// Returns a sync callback that wraps this function for tool execution
            #fn_vis fn #callback_fn_name() -> std::sync::Arc<mistralrs::ToolCallback> {
                std::sync::Arc::new(|called: &mistralrs::CalledFunction| {
                    let args: #args_struct_name = serde_json::from_str(&called.arguments)
                        .map_err(|e| anyhow::anyhow!("Failed to parse tool arguments: {}", e))?;

                    let result = #fn_name(#(#call_args),*)?;

                    serde_json::to_string(&result)
                        .map_err(|e| anyhow::anyhow!("Failed to serialize tool result: {}", e))
                })
            }

            /// Returns both the Tool definition and callback as a tuple
            #fn_vis fn #combined_fn_name() -> (mistralrs::Tool, mistralrs::ToolCallbackType) {
                (#tool_fn_name(), mistralrs::ToolCallbackType::Sync(#callback_fn_name()))
            }
        }
    };

    Ok(output)
}

/// Check if a type is Option<T>
fn is_option_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            return segment.ident == "Option";
        }
    }
    false
}

/// Extract the inner type from Option<T>, returning None if not an Option
fn extract_option_inner_type(ty: &Type) -> Option<&Type> {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            if segment.ident == "Option" {
                if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                    if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                        return Some(inner);
                    }
                }
            }
        }
    }
    None
}
