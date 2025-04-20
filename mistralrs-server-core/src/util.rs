use image::DynamicImage;
use tokio::{
    fs::{self, File},
    io::AsyncReadExt,
};

pub async fn parse_image_url(url_unparsed: &str) -> Result<DynamicImage, anyhow::Error> {
    let url = if let Ok(url) = url::Url::parse(url_unparsed) {
        url
    } else if File::open(url_unparsed).await.is_ok() {
        url::Url::from_file_path(std::path::absolute(url_unparsed)?)
            .map_err(|_| anyhow::anyhow!("Could not parse file path: {}", url_unparsed))?
    } else {
        url::Url::parse(url_unparsed)
            .map_err(|_| anyhow::anyhow!("Could not parse as base64 data: {}", url_unparsed))?
    };

    let bytes = if url.scheme() == "http" || url.scheme() == "https" {
        // Read from http
        match reqwest::get(url.clone()).await {
            Ok(http_resp) => http_resp.bytes().await?.to_vec(),
            Err(e) => anyhow::bail!(e),
        }
    } else if url.scheme() == "file" {
        let path = url
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Could not parse file path: {}", url))?;

        if let Ok(mut f) = File::open(&path).await {
            // Read from local file
            let metadata = fs::metadata(&path).await?;
            let mut buffer = vec![0; metadata.len() as usize];
            f.read_exact(&mut buffer).await?;
            buffer
        } else {
            anyhow::bail!("Could not open file at path: {}", url);
        }
    } else if url.scheme() == "data" {
        // Decode with base64
        let data_url = data_url::DataUrl::process(url.as_str())?;
        data_url.decode_to_vec()?.0
    } else {
        anyhow::bail!("Unsupported URL scheme: {}", url.scheme());
    };

    Ok(image::load_from_memory(&bytes)?)
}

#[cfg(test)]
mod tests {
    use image::GenericImageView;

    use super::*;

    #[tokio::test]
    async fn test_parse_image_url() {
        // from URL
        let url = "https://www.rust-lang.org/logos/rust-logo-32x32.png";
        let image = parse_image_url(url).await.unwrap();
        assert_eq!(image.dimensions(), (32, 32));

        let url = "http://www.rust-lang.org/logos/rust-logo-32x32.png";
        let image = parse_image_url(url).await.unwrap();
        assert_eq!(image.dimensions(), (32, 32));

        // from file path
        let url = "resources/rust-logo-32x32.png";
        let image = parse_image_url(url).await.unwrap();
        assert_eq!(image.dimensions(), (32, 32));

        // URL must be an absolute path
        let absolute_path = std::path::absolute(url).unwrap();
        let url = format!("file://{}", absolute_path.as_os_str().to_str().unwrap());
        let image = parse_image_url(&url).await.unwrap();
        assert_eq!(image.dimensions(), (32, 32));

        // from base64 encoded image (rust-logo-32x32.png)
        let url = "
        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
        ";
        let image = parse_image_url(url).await.unwrap();
        assert_eq!(image.dimensions(), (32, 32));

        let url = format!("data:image/png;base64,{}", url);
        let image = parse_image_url(&url).await.unwrap();
        assert_eq!(image.dimensions(), (32, 32));
    }
}
