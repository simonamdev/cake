use std::{collections::HashMap, fs};

use serde_json::Value;
use sha2::{Digest, Sha256};

use reqwest::blocking::Client;

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
    pub header_length_bytes: u64,
}

pub fn get_locally_available_hashes(storage_dir: &str) -> Vec<String> {
    // If storage dir does not exist, create it first
    if fs::metadata(storage_dir).is_err() {
        fs::create_dir_all(storage_dir).expect("Failed to create storage directory");
    }
    // Search through available hases
    let locally_available_hashes: Vec<String> = fs::read_dir(storage_dir)
        .unwrap()
        .filter_map(|entry| {
            entry
                .ok()
                .and_then(|dir_entry| dir_entry.file_name().into_string().ok())
        })
        .map(|entry| entry.to_string())
        .collect();

    // println!("{}", locally_available_hashes[0]);

    // println!(
    //     "{} layers already available locally",
    //     locally_available_hashes.len()
    // );

    locally_available_hashes
}

pub fn get_model_file_hashes(
    model_id: &str,
    file_name: &str,
) -> (ModelHeader, HashMap<String, String>) {
    let client = Client::new();

    // TODO: Pass this as an env var
    // TODO: Support custom base URLs
    let registry_base_url = "http://localhost:3000";

    let model_hashes_url = format!("{}/results/{}/hashes.json", registry_base_url, model_id);

    // TODO: Error handling
    // TODO: Handle the situation where the registry is unavailable by downloading all of the layers
    let response = client.get(model_hashes_url).send().unwrap();

    let hashes: Value = response.json().unwrap();
    let mut layer_to_hash_map: HashMap<String, String> = HashMap::new();

    for (key, value) in hashes.as_object().unwrap() {
        // println!("{}", value);
        if file_name == value.get("file_name").unwrap().as_str().unwrap() {
            layer_to_hash_map.insert(
                key.to_string(),
                value.get("hash").unwrap().as_str().unwrap().to_string(),
            );
        }
    }

    let model_file_url = &download::get_download_url_from_model_id(model_id, file_name);

    // Download the header to understand the file
    // TODO: This could be retrieved and cached by the registry
    println!("Retrieving header for {}: {}", model_id, file_name);
    let (header, header_length) = download::download_safetensors_header(model_file_url);

    (
        ModelHeader {
            raw_header: header,
            header_length_bytes: header_length,
        },
        layer_to_hash_map,
    )
}
