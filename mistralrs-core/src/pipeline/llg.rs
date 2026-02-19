use std::sync::Arc;

use anyhow::Result;
use llguidance::{api::TopLevelGrammar, ParserFactory};
use tokenizers::Tokenizer;

use crate::Constraint;

pub fn build_llg_factory(tokenizer: Tokenizer) -> Result<Arc<ParserFactory>> {
    // Collect special token info before from_tokenizer() consumes the tokenizer.
    let added_special: Vec<(u32, String)> = tokenizer
        .get_added_tokens_decoder()
        .into_iter()
        .filter(|(_, at)| at.special)
        .map(|(id, at)| (id, at.content))
        .collect();

    let bt = toktrie_hf_tokenizers::ByteTokenizer::from_tokenizer(tokenizer)?;
    let info = bt.tokrx_info();
    let mut token_bytes = bt.token_bytes();

    // Fix up special tokens that from_tokenizer() may have missed.
    // Tekken tokenizers add special tokens to the BPE base vocab first, which can
    // prevent the toktrie builder from applying the SPECIAL_TOKEN_MARKER prefix.
    for (id, content) in &added_special {
        let idx = *id as usize;
        if idx < token_bytes.len()
            && (token_bytes[idx].is_empty()
                || token_bytes[idx][0] != toktrie::TokTrie::SPECIAL_TOKEN_MARKER)
        {
            let mut bytes = content.as_bytes().to_vec();
            bytes.insert(0, toktrie::TokTrie::SPECIAL_TOKEN_MARKER);
            token_bytes[idx] = bytes;
        }
    }

    let tok_trie = toktrie::TokTrie::from(&info, &token_bytes);
    let env = toktrie_hf_tokenizers::ByteTokenizerEnv {
        tokenizer: bt,
        tok_trie,
    }
    .to_env();
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
