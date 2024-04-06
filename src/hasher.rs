use std::collections::HashMap;

use sha2::{Digest, Sha256};

use crate::download;

pub fn sha256_hash(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let result = hasher.finalize();

    let hash_hex = format!("{:x}", result);

    hash_hex
}

pub struct ModelHeader {
    pub raw_header: serde_json::Value,
    header_length_bytes: u64,
}

pub fn get_locally_available_hashes(required_hashes: Vec<String>) -> Vec<String> {
    // TODO: Implement this check locally
    let locally_available_hashes: Vec<String> = vec![];

    locally_available_hashes
}

pub fn get_model_file_hashes(
    model_id: &str,
    file_name: &str,
) -> (ModelHeader, HashMap<String, String>) {
    // TODO: Implement this retrieval from the registry
    let mut layer_to_hash_map = HashMap::new();
    layer_to_hash_map.insert(
        "fake-layer-name".to_string(),
        "fake-layer-hash-abc123".to_string(),
    );

    let file_url = &download::get_download_url_from_model_id(model_id, file_name);

    // Download the header to understand the file
    // TODO: This could be retrieved and cached by the registry
    println!("Retrieving header for {}: {}", model_id, file_name);
    let (header, header_length) = download::download_safetensors_header(file_url);

    (
        ModelHeader {
            raw_header: header,
            header_length_bytes: header_length,
        },
        layer_to_hash_map,
    )
}
