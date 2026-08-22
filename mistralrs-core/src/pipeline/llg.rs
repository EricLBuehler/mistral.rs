use std::sync::Arc;

use anyhow::Result;
use llguidance::{api::TopLevelGrammar, ParserFactory};
use tokenizers::{decoders::DecoderWrapper, Tokenizer};

use crate::Constraint;

pub fn build_llg_factory(mut tokenizer: Tokenizer) -> Result<Arc<ParserFactory>> {
    let decoder = match tokenizer.get_decoder() {
        Some(DecoderWrapper::Sequence(sequence)) if sequence.get_decoders().len() == 1 => {
            match &sequence.get_decoders()[0] {
                DecoderWrapper::ByteLevel(decoder) => Some(DecoderWrapper::ByteLevel(*decoder)),
                _ => None,
            }
        }
        _ => None,
    };
    if let Some(decoder) = decoder {
        tokenizer.with_decoder(Some(decoder));
    }

    // Collect special token info before from_tokenizer() consumes the tokenizer.
    let added_special: Vec<(u32, String)> = tokenizer
        .get_added_tokens_decoder()
        .into_iter()
        .filter(|(_, at)| at.special)
        .map(|(id, at)| (id, at.content))
        .collect();

    // toktrie_hf_tokenizers marks every added-vocab token special; honor the tokenizer's own
    // `special` flag so non-special added tokens (e.g. PaddleOCR-VL OTSL <fcel>/<nl>) decode as
    // content, matching transformers decode(skip_special_tokens=True) instead of being dropped.
    let added_nonspecial: Vec<u32> = tokenizer
        .get_added_tokens_decoder()
        .into_iter()
        .filter(|(_, at)| !at.special)
        .map(|(id, _)| id)
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

    for id in &added_nonspecial {
        let idx = *id as usize;
        if idx < token_bytes.len()
            && token_bytes[idx].first() == Some(&toktrie::TokTrie::SPECIAL_TOKEN_MARKER)
        {
            token_bytes[idx].remove(0);
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

#[cfg(test)]
mod tests {
    use super::*;

    // Mirrors build_llg_factory's toktrie assembly; `apply_fix=false` reproduces the pre-fix
    // behavior (every added-vocab token stays special-marked), `true` honors the tokenizer's own
    // `special` flag. Returns the trie so decode_ext can be asserted directly.
    fn build_trie(mut tokenizer: Tokenizer, apply_fix: bool) -> toktrie::TokTrie {
        let decoder = match tokenizer.get_decoder() {
            Some(DecoderWrapper::Sequence(sequence)) if sequence.get_decoders().len() == 1 => {
                match &sequence.get_decoders()[0] {
                    DecoderWrapper::ByteLevel(decoder) => Some(DecoderWrapper::ByteLevel(*decoder)),
                    _ => None,
                }
            }
            _ => None,
        };
        if let Some(decoder) = decoder {
            tokenizer.with_decoder(Some(decoder));
        }
        let added_special: Vec<(u32, String)> = tokenizer
            .get_added_tokens_decoder()
            .into_iter()
            .filter(|(_, at)| at.special)
            .map(|(id, at)| (id, at.content))
            .collect();
        let added_nonspecial: Vec<u32> = tokenizer
            .get_added_tokens_decoder()
            .into_iter()
            .filter(|(_, at)| !at.special)
            .map(|(id, _)| id)
            .collect();
        let bt = toktrie_hf_tokenizers::ByteTokenizer::from_tokenizer(tokenizer).unwrap();
        let info = bt.tokrx_info();
        let mut token_bytes = bt.token_bytes();
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
        if apply_fix {
            for id in &added_nonspecial {
                let idx = *id as usize;
                if idx < token_bytes.len()
                    && token_bytes[idx].first() == Some(&toktrie::TokTrie::SPECIAL_TOKEN_MARKER)
                {
                    token_bytes[idx].remove(0);
                }
            }
        }
        toktrie::TokTrie::from(&info, &token_bytes)
    }

    fn dec(trie: &toktrie::TokTrie, id: u32) -> String {
        String::from_utf8_lossy(&trie.decode_ext(&[id], false)).into_owned()
    }

    // Env-gated on local tokenizer files (skips in CI). Proves: non-special added tokens now decode
    // as content, while special=true tokens (EOS, chat delimiters, image) stay dropped as before.
    #[test]
    fn honors_tokenizer_special_flag() {
        if let Ok(p) = std::env::var("REGRESSION_PADDLE_TOK") {
            let tok = Tokenizer::from_file(&p).unwrap();
            let fcel = tok.token_to_id("<fcel>").unwrap();
            let eos = tok.token_to_id("</s>").unwrap();
            let imend = tok.token_to_id("<|IMAGE_END|>").unwrap();
            let (before, after) = (build_trie(tok.clone(), false), build_trie(tok, true));
            println!(
                "paddle <fcel>  before={:?} after={:?}",
                dec(&before, fcel),
                dec(&after, fcel)
            );
            println!(
                "paddle </s>    before={:?} after={:?}",
                dec(&before, eos),
                dec(&after, eos)
            );
            println!(
                "paddle IMG_END before={:?} after={:?}",
                dec(&before, imend),
                dec(&after, imend)
            );
            assert!(
                dec(&before, fcel).is_empty(),
                "pre-fix drops the non-special OTSL token"
            );
            assert_eq!(
                dec(&after, fcel),
                "<fcel>",
                "fix keeps non-special content token"
            );
            assert!(
                dec(&after, eos).is_empty(),
                "special=true </s> still dropped"
            );
            assert!(
                dec(&after, imend).is_empty(),
                "special=true image token still dropped"
            );
        }
        if let Ok(p) = std::env::var("REGRESSION_QWEN_TOK") {
            let tok = Tokenizer::from_file(&p).unwrap();
            let im_start = tok.token_to_id("<|im_start|>").unwrap();
            let im_end = tok.token_to_id("<|im_end|>").unwrap();
            let tool = tok.token_to_id("<tool_call>").unwrap();
            let (before, after) = (build_trie(tok.clone(), false), build_trie(tok, true));
            println!(
                "qwen im_start  before={:?} after={:?}",
                dec(&before, im_start),
                dec(&after, im_start)
            );
            println!(
                "qwen tool_call before={:?} after={:?}",
                dec(&before, tool),
                dec(&after, tool)
            );
            assert!(
                dec(&before, im_start).is_empty() && dec(&after, im_start).is_empty(),
                "chat delimiter <|im_start|> (special=true) stays dropped before AND after"
            );
            assert!(
                dec(&after, im_end).is_empty(),
                "chat delimiter <|im_end|> stays dropped"
            );
        }
    }

    // Self-contained (runs in CI, no external files): a byte-fallback tokenizer with a special=false
    // <x> and a special=true <s>. Both are <...>-shaped, so from_tokenizer's heuristic marks both;
    // pre-fix drops <x>, the fix keeps it as content while <s> stays dropped.
    #[test]
    fn honors_special_flag_inline_fixture() {
        use tokenizers::{
            decoders::byte_level::ByteLevel as ByteLevelDecoder, models::bpe::BpeBuilder,
            AddedToken,
        };
        let vocab = ahash::AHashMap::from([("a".to_string(), 0u32)]);
        let bpe = BpeBuilder::new()
            .vocab_and_merges(vocab, vec![])
            .build()
            .unwrap();
        let mut tok = Tokenizer::new(bpe);
        tok.with_decoder(Some(ByteLevelDecoder::new(true, false, false)));
        tok.add_tokens(&[
            AddedToken::from("<x>", false),
            AddedToken::from("<s>", true),
        ]);
        let x = tok.token_to_id("<x>").unwrap();
        let s = tok.token_to_id("<s>").unwrap();
        let before = build_trie(tok.clone(), false);
        let after = build_trie(tok, true);
        assert!(
            dec(&before, x).is_empty(),
            "pre-fix: <...> heuristic drops the special=false token"
        );
        assert_eq!(
            dec(&after, x),
            "<x>",
            "fix keeps the special=false content token"
        );
        assert!(
            dec(&after, s).is_empty(),
            "special=true token stays dropped"
        );
    }
}
