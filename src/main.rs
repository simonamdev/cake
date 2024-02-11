use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufReader, ErrorKind, Read, Seek, SeekFrom};
use std::convert::TryInto;
use std::path::{Path, PathBuf};

use serde_json::{json, value, Error, Value};
use sha2::{Sha256, Digest};

fn main() {
    // let file_path = "/home/simon/Downloads/models/mistral-7B-v0.1/model-00001-of-00002.safetensors";
    // let hashed_model_result = hash_safetensors_file(file_path);
    // match hashed_model_result {
    //     Ok(value) => {
    //         println!("{}", value);
    //     }
    //     Err(e) => {
    //         eprintln!("{}", e);
    //     }
    // }

    let root_models_dir = "/media/simon/models/results";
    let model_directories = get_models_by_account(root_models_dir);
    for (key, value) in model_directories {
        println!("{}, {:?}", key, value);
        for dir in value {
            let model_safetensors = get_model_safetensor_files(&dir);
            println!("{:?}", model_safetensors);
        }
    }
}

fn get_model_safetensor_files(directory: &str) -> Vec<PathBuf> {
    let mut safetensor_files: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = fs::read_dir(directory) {
        for entry in entries {
            if let Ok(entry) = entry {
                let file_path = entry.path();

                if let Some(extension) = file_path.extension() {
                    if extension == "safetensors" {
                        safetensor_files.push(file_path);
                    }
                }
            }
        }
    }

    return safetensor_files
}

fn get_models_by_account(root_models_directory: &str) -> HashMap<String, Vec<String>> {
    let mut models_by_account_map: HashMap<String, Vec<String>> = HashMap::new();

    // First set of directories is the HF account
    let owners = get_directories(root_models_directory);

    owners.into_iter().for_each(|owner| {
        let owner_clone = owner.clone();
        let mut model_dir = PathBuf::from(root_models_directory);
        model_dir.push(owner);
        let mut models = get_directories(&model_dir.to_string_lossy().into_owned());
        let entry = models_by_account_map.entry(owner_clone).or_insert_with(Vec::new);
        entry.append(&mut models);
    });

    models_by_account_map
}

fn get_directories(directory: &str) -> Vec<String> {
    let mut model_directories: Vec<String> = Vec::new();

    if let Ok(entries) = fs::read_dir(directory) {
        for entry in entries {
            if let Ok(entry) = entry {
                let file_path = entry.path();

                if file_path.is_dir() {
                    // Get the full path
                    println!("{}", file_path.display());
                    if let Some(folder_name) = file_path.file_name() {
                        model_directories.push(file_path.to_string_lossy().into_owned());
                    }
                }
            }
        }
    }

    return model_directories
}

fn hash_safetensors_file(file_path: &str) -> Result<Value, io::Error> {
    let mut file = BufReader::new(File::open(file_path)?);

    // Read the JSON header length
    let mut buffer = [0; 8];
    file.read_exact(&mut buffer)?;
    let json_header_length = u64::from_le_bytes(buffer);

    println!("JSON Header Length: {}", json_header_length);

    // Skip the first 8 bytes (header)
    file.seek(SeekFrom::Start(8))?;

    // Read the JSON header into a buffer
    let mut json_buffer = vec![0; json_header_length.try_into().unwrap()];
    file.read_exact(&mut json_buffer)?;

    // Switch the buffer to a string
    let json_string = String::from_utf8_lossy(&json_buffer);
    println!("{}", json_string);

    let json_value: Result<Value, _> = serde_json::from_str(&json_string);

    let mut output_object = json!({
        "file_path": file_path,
        "tensors": {}
    });

    match json_value {
        Ok(value) => {
            if let Value::Object(map) = value {
         
                for (index, (key, value)) in map.iter().enumerate() {
                    println!("Key: {}, Value: {}", key, value);
                    if key != "__metadata__" {
                        // Get the data_offsets
                        let offsets = value.get("data_offsets").and_then(Value::as_array).unwrap();
                        let offset_start = offsets[0].as_u64().unwrap();
                        let offset_end = offsets[1].as_u64().unwrap();
                        let offset_diff = offset_end - offset_start;
                        println!("Offset Difference: {:?}", offset_diff);
        
                        println!("Seeking to data");
                        // Seek to the start position of the tensor data
                        file.seek(SeekFrom::Start(offset_start))?;
        
                        // Read tensor data into buffer
                        println!("Reading into buffer");
                        let mut tensor_buffer = vec![0; (offset_diff / 8).try_into().unwrap()];
                        file.read_exact(&mut tensor_buffer)?;
        
                        // Calculate SHA-256 hash of tensor data
                        println!("{} / {} Hashing...", index, map.len());
                        let hash = sha256_hash(&tensor_buffer);
                        println!("SHA-256 Hash: {}", hash);
                        output_object["layers"][key] = json!(hash);
                    }
                }
            }
        }
        Err(e) => eprintln!("Error deserailizing JSON: {}", e)
    }

    Ok(output_object)
}

fn sha256_hash(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let result = hasher.finalize();

    let hash_hex = format!("{:x}", result);

    hash_hex
}