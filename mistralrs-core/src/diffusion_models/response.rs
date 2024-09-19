use std::io::Cursor;

use base64::{engine::general_purpose::STANDARD, Engine};
use image::DynamicImage;
use uuid::Uuid;

use crate::{
    sequence::{Sequence, SequenceState, StopReason},
    ImageChoice, ImageGenerationResponse, ImageGenerationResponseFormat,
};

pub async fn send_responses(
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
                let saved_path = format!("image-generation-{}.png", Uuid::new_v4());
                image
                    .save_with_format(&saved_path, image::ImageFormat::Png)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                ImageChoice {
                    url: Some(saved_path),
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
