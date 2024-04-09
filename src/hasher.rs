use std::{collections::HashMap, fs};

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

pub fn get_locally_available_hashes(storage_dir: &str) -> Vec<String> {
    let locally_available_hashes: Vec<String> = fs::read_dir(storage_dir)
        .unwrap()
        .filter_map(|entry| {
            entry.ok().and_then(|dir_entry| {
                dir_entry.file_name().into_string().ok()
            })
        })
        .collect();

    println!("{} layers already available locally", locally_available_hashes.len());

    locally_available_hashes
}

pub fn get_model_file_hashes(
    model_id: &str,
    file_name: &str,
) -> (ModelHeader, HashMap<String, String>) {
    // TODO: Implement this retrieval from the registry
    let mut layer_to_hash_map = HashMap::new();
    // layer_to_hash_map.insert(
    //     "fake-layer-name".to_string(),
    //     "fake-layer-hash-abc123".to_string(),
    // );

    // KoboldAI/fairseq-dense-1.3B: lm_head.weight
    layer_to_hash_map.insert(
        "lm_head.weight".to_string(),
        "699a0dd9f0ce1218da2b7fbc61d73dfd922595f4cbf573e5bc222a0991d08c18".to_string(),
    );
    layer_to_hash_map.insert(
        "model.layers.9.self_attn_layer_norm.weight".to_string(),
        "5998cac70b9ca80e85f3404cddd785c52701743a80fdc322df947b52071fb55a".to_string(),
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
