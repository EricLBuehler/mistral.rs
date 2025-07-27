use std::sync::Arc;

use anyhow::Result;
use llguidance::{api::TopLevelGrammar, ParserFactory};

use crate::tokenizer::TokenizerImpl;
use crate::Constraint;

pub fn build_llg_factory(tokenizer: TokenizerImpl) -> Result<Arc<ParserFactory>> {
    // Get the base tokenizer - all variants have the same underlying tokenizer
    let base_tokenizer = tokenizer.get_base_tokenizer().clone();
    let env =
        toktrie_hf_tokenizers::ByteTokenizer::from_tokenizer(base_tokenizer)?.into_tok_env(None)?;
    let factory = ParserFactory::new_simple(&env)?;
    Ok(Arc::new(factory))
}

pub fn llg_grammar_from_constraint(constraint: &Constraint) -> Result<Option<TopLevelGrammar>> {
    let grm = match constraint {
        Constraint::Regex(regex) => TopLevelGrammar::from_regex(regex),
        Constraint::Lark(lark) => TopLevelGrammar::from_lark(lark.clone()),
        Constraint::JsonSchema(value) => TopLevelGrammar::from_json_schema(value.clone()),
        Constraint::Llguidance(value) => value.clone(),
        Constraint::None => return Ok(None),
    };
    Ok(Some(grm))
}

pub fn constraint_from_llg_grammar(
    factory: &ParserFactory,
    grm: TopLevelGrammar,
) -> Result<llguidance::Matcher> {
    let parser = factory.create_parser(grm)?;
    Ok(llguidance::Matcher::new(Ok(parser)))
}
