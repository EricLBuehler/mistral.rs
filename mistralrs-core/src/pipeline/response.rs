use std::{io::Cursor, sync::Arc};

use base64::{engine::general_purpose::STANDARD, Engine};
use candle_core::Tensor;
use image::DynamicImage;
use uuid::Uuid;

use crate::{
    sequence::{Sequence, SequenceState, StopReason},
    ImageChoice, ImageGenerationResponse, ImageGenerationResponseFormat,
};

pub async fn send_image_responses(
    input_seqs: &mut [&mut Sequence],
    images: Vec<DynamicImage>,
) -> candle_core::Result<()> {
    if input_seqs.len() != images.len() {
        candle_core::bail!(
            "Input seqs len ({}) does not match images generated len ({})",
            input_seqs.len(),
            images.len()
        );
    }

    for (seq, image) in input_seqs.iter_mut().zip(images) {
        let choice = match seq
            .image_gen_response_format()
            .unwrap_or(ImageGenerationResponseFormat::Url)
        {
            ImageGenerationResponseFormat::Url => {
                let saved_file = match seq.image_gen_save_file() {
                    Some(path) => path.to_string_lossy().into_owned(),
                    None => format!("image-generation-{}.png", Uuid::new_v4()),
                };
                image
                    .save_with_format(&saved_file, image::ImageFormat::Png)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                ImageChoice {
                    url: Some(saved_file),
                    b64_json: None,
                }
            }
            ImageGenerationResponseFormat::B64Json => {
                let mut buffer = Vec::new();
                image
                    .write_to(&mut Cursor::new(&mut buffer), image::ImageFormat::Png)
                    .expect("Failed to encode image");
                let encoded = STANDARD.encode(&buffer);
                let serialized_b64 = format!("data:image/png;base64,{encoded}");
                ImageChoice {
                    url: None,
                    b64_json: Some(serialized_b64),
                }
            }
        };
        seq.add_image_choice_to_group(choice);

        let group = seq.get_mut_group();
        group
            .maybe_send_image_gen_response(
                ImageGenerationResponse {
                    created: seq.creation_time() as u128,
                    data: group.get_image_choices().to_vec(),
                },
                seq.responder(),
            )
            .await
            .map_err(candle_core::Error::msg)?;

        seq.set_state(SequenceState::Done(StopReason::GeneratedImage));
    }

    Ok(())
}

pub async fn send_speech_responses(
    input_seqs: &mut [&mut Sequence],
    pcms: &[Arc<Vec<f32>>],
    rates: &[usize],
    channels: &[usize],
) -> candle_core::Result<()> {
    if input_seqs.len() != pcms.len() {
        candle_core::bail!(
            "Input seqs len ({}) does not match pcms generated len ({})",
            input_seqs.len(),
            pcms.len()
        );
    }

    for (seq, (pcm, (rate, channel))) in input_seqs
        .iter_mut()
        .zip(pcms.iter().zip(rates.iter().zip(channels)))
    {
        seq.add_speech_pcm_to_group(pcm.clone(), *rate, *channel);

        let group = seq.get_mut_group();
        group
            .maybe_send_speech_response(seq.responder())
            .await
            .map_err(candle_core::Error::msg)?;

        seq.set_state(SequenceState::Done(StopReason::GeneratedSpeech));
    }

    Ok(())
}

pub async fn send_raw_responses(
    input_seqs: &mut [&mut Sequence],
    logits_chunks: Vec<Vec<Tensor>>,
) -> candle_core::Result<()> {
    let logits_chunks = if logits_chunks.len() == 1 {
        logits_chunks[0].clone()
    } else {
        candle_core::bail!("Raw response only supports batch size of 1.");
    };
    assert_eq!(input_seqs.len(), 1);

    let seq = &mut *input_seqs[0];

    seq.add_raw_choice_to_group(logits_chunks);

    let group = seq.get_mut_group();
    group
        .maybe_send_raw_done_response(seq.responder())
        .await
        .map_err(candle_core::Error::msg)?;

    seq.set_state(SequenceState::Done(StopReason::Length(0)));

    Ok(())
}

pub async fn send_embedding_responses(
    input_seqs: &mut [&mut Sequence],
    embedings: Vec<Vec<f32>>,
) -> candle_core::Result<()> {
    if embedings.len() != input_seqs.len() {
        candle_core::bail!("Number of embeddings must match number of sequences..");
    }

    for (seq, embeddings) in input_seqs.iter_mut().zip(embedings) {
        seq.add_embedding_choice_to_group(embeddings);

        let group = seq.get_mut_group();
        group
            .maybe_send_embedding_done_response(seq.responder())
            .await
            .map_err(candle_core::Error::msg)?;

        seq.set_state(SequenceState::Done(StopReason::Length(0)));
    }

    Ok(())
}
