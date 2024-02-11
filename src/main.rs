use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{self, BufReader, ErrorKind, Read, Seek, SeekFrom, Write};
use std::convert::TryInto;
use std::path::{Path, PathBuf};

use serde_json::{json, value, Error, Map, Value};
use sha2::{Sha256, Digest};

use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};

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

    let root_models_dir = "/media/simon/models3/results";
    let model_directories_by_account = get_models_by_account(root_models_dir);

    let account_count = model_directories_by_account.keys().len();
    
    let bar = ProgressBar::new(account_count.try_into().unwrap());

    for (model_account, model_dirs) in model_directories_by_account {
        bar.inc(1);
        // println!("{}/{} {}, {:?}", i, account_count, model_account, model_dirs);

        for model_name_dir in model_dirs {
            let model_safetensors = get_model_safetensor_files(&model_name_dir);
            let model_name = model_name_dir.split("/").last().unwrap();
            let mut hashed_model_safetensors: Vec<Value> = Vec::new();
            println!("{:?}", model_safetensors);
            for safetensors_file_path in model_safetensors {
                let hashed_model_result = hash_safetensors_file(&safetensors_file_path.to_string_lossy().into_owned());
                match hashed_model_result {
                    Ok(value) => {
                        // println!("{}", value);
                        hashed_model_safetensors.push(value);
                    }
                    Err(e) => {
                        eprintln!("{}", e);
                    }
                }
            }
            let final_output_json = merge_hash_results(hashed_model_safetensors);
            // println!("{:?}", final_output_json);
            // Form the file path for the results
            let model_account_clone = model_account.clone();
            // Forgive me for this but pathbuf wasn't funcitoning as expected
            let abs_dir: String =  env::current_dir().unwrap().to_string_lossy().into_owned().clone() + &"/results/".to_owned() + &model_account_clone.clone() + "/" + &model_name;
            // Build the dir in case it doesnt exist
            let abs_dir_clone: String = abs_dir.clone();
            fs::create_dir_all(abs_dir).unwrap();
            let target_file_path = abs_dir_clone.to_owned() + "/hashes.json";

            write_json_to_file(final_output_json, target_file_path).unwrap();
        }
    }
    bar.finish();
}

fn write_json_to_file(data: Value, file_path: String) -> io::Result<()> {
    println!("Writing to: {:?}", file_path);
    let json_string = serde_json::to_string_pretty(&data)?;
    let mut file = File::create(file_path)?;
    file.write_all(json_string.as_bytes())?;
    Ok(())
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
    let owners_dirs = get_directories(root_models_directory);

    owners_dirs.into_iter()
        .map(|owner_dir| owner_dir.clone().split("/").last().unwrap().to_string())
        .for_each(|owner| {
            let owner_clone = owner.clone();
            let mut model_dir = PathBuf::from(root_models_directory);
            model_dir.push(owner);
            let mut models = get_directories(&model_dir.to_string_lossy().into_owned());
            let entry = models_by_account_map.entry(owner_clone.to_string()).or_insert_with(Vec::new);
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

fn merge_hash_results(json_results: Vec<Value>) -> Value {
    // Setup new variables to hold the merged values
    let mut file_paths: HashSet<String> = HashSet::new();
    let mut merged_tensors: Map<String, Value> = Map::new();

    for json_result in json_results {
        let file_path = json_result.get("file_path").and_then(Value::as_str).unwrap();
        file_paths.insert(file_path.to_string());
        let tensors = json_result.get("tensors").and_then(Value::as_object).unwrap();
        for (key, value) in tensors {
            merged_tensors.entry(key.clone()).or_insert_with(|| value.clone());
        }
    }

    let merged_json = json!({
        "file_paths": file_paths,
        "tensors": merged_tensors
    });

    return merged_json
}

fn hash_safetensors_file(file_path: &str) -> Result<Value, io::Error> {
    let mut file = BufReader::new(File::open(file_path)?);

    // Read the JSON header length
    let mut buffer = [0; 8];
    file.read_exact(&mut buffer)?;
    let json_header_length = u64::from_le_bytes(buffer);

    // println!("JSON Header Length: {}", json_header_length);

    // Skip the first 8 bytes (header)
    file.seek(SeekFrom::Start(8))?;

    // Read the JSON header into a buffer
    let mut json_buffer = vec![0; json_header_length.try_into().unwrap()];
    file.read_exact(&mut json_buffer)?;

    // Switch the buffer to a string
    let json_string = String::from_utf8_lossy(&json_buffer);
    // println!("{}", json_string);

    let json_value: Result<Value, _> = serde_json::from_str(&json_string);

    let mut output_object = json!({
        "file_path": file_path,
        "tensors": {}
    });

    match json_value {
        Ok(value) => {
            if let Value::Object(map) = value {
         
                for (index, (key, value)) in map.iter().enumerate() {
                    // println!("Key: {}, Value: {}", key, value);
                    if key != "__metadata__" {
                        // Get the data_offsets
                        let offsets = value.get("data_offsets").and_then(Value::as_array).unwrap();
                        let offset_start = offsets[0].as_u64().unwrap();
                        let offset_end = offsets[1].as_u64().unwrap();
                        let offset_diff = offset_end - offset_start;
                        // println!("Offset Difference: {:?}", offset_diff);
        
                        // println!("Seeking to data");
                        // Seek to the start position of the tensor data
                        file.seek(SeekFrom::Start(offset_start))?;
        
                        // Read tensor data into buffer
                        // println!("Reading into buffer");
                        let mut tensor_buffer = vec![0; (offset_diff / 8).try_into().unwrap()];
                        file.read_exact(&mut tensor_buffer)?;
        
                        // Calculate SHA-256 hash of tensor data
                        // println!("{} / {} Hashing...", index+1, map.len());
                        let hash = sha256_hash(&tensor_buffer);
                        // println!("SHA-256 Hash: {}", hash);
                        output_object["tensors"][key] = json!(hash);
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