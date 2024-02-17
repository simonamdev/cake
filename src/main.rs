use core::panic;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{self, BufReader, ErrorKind, Read, Seek, SeekFrom, Write};
use std::convert::TryInto;
use std::path::{Path, PathBuf};

use safetensors::{SafeTensorError, SafeTensors};
use serde_json::{json, value, Error, Map, Value};
use sha2::{Sha256, Digest};

use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};

mod download;

fn main() {
    let url = "https://huggingface.co/KoboldAI/fairseq-dense-1.3B/resolve/main/model.safetensors?download=true";
    let download_folder = "./download";
    let cache_folder = "./cache";
    download::download_full_safetensors_file(url, download_folder, cache_folder);
    let target_file_path = "./test.safetensors";
    download::combine_cached_files_to_safetensors_file(cache_folder, target_file_path);
    // Test if it deserialises properly
    let mut bytes = vec![];
    let mut f = fs::File::open(target_file_path).unwrap();
    f.read_to_end(&mut bytes).unwrap();
    let result = SafeTensors::deserialize(&bytes).unwrap();
    // process_files_locally();
}

fn process_files_locally() {
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

    let root_models_dir = "/media/simon/models2/results";
    let model_directories_by_account = get_models_by_account(root_models_dir);

    let account_count = model_directories_by_account.len();
    println!("{} Accounts found...", account_count);
    
    let bar = ProgressBar::new(account_count.try_into().unwrap());
    bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {percent}% {msg}")
            .unwrap()
            .progress_chars("##-"),
    );
    for (model_account, model_dirs) in model_directories_by_account {
        // println!("{}/{} {}, {:?}", i, account_count, model_account, model_dirs);
        bar.inc(1);

        for model_name_dir in model_dirs {
            let model_safetensors = get_model_safetensor_files(&model_name_dir);
            let model_name = model_name_dir.split("/").last().unwrap();
            bar.set_message(format!("{}/{}", model_account, model_name));

            let target_dir_and_path = get_hashes_file_dir_and_path(&model_account, model_name);
            // If the target file already exists, then skip rehashing
            // TODO: Override flag to force recreating it?
            let target_dir_and_path_clone = target_dir_and_path.clone();
            let file_already_exists = fs::metadata(target_dir_and_path.1).is_ok();

            if file_already_exists {
                continue
            }

            let mut hashed_model_safetensors: Vec<Value> = Vec::new();
            // println!("{:?}", model_safetensors);
            for safetensors_file_path in model_safetensors {
                let hashed_model_result = hash_safetensors_file(&safetensors_file_path.to_string_lossy().into_owned());
                match hashed_model_result {
                    Ok(value) => {
                        // println!("{}", value);
                        hashed_model_safetensors.push(value);
                    }
                    Err(e) => {
                        panic!("{}", format!("{}", e));
                    }
                }
            }
            let final_output_json = merge_hash_results(hashed_model_safetensors);
            // println!("{:?}", final_output_json);
            
            fs::create_dir_all(target_dir_and_path.0).unwrap();
            write_json_to_file(final_output_json, target_dir_and_path_clone.1).unwrap();
        }
    }
    bar.finish();
}

fn get_hashes_file_dir_and_path(model_account: &str, model_name: &str) -> (String, String) {
    let abs_dir: String =  env::current_dir().unwrap().to_string_lossy().into_owned().clone() + &"/results/".to_owned() + &model_account + "/" + &model_name;
    let target_file_path = abs_dir.to_owned() + "/hashes.json";

    (abs_dir, target_file_path)
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

                let progress_bar = ProgressBar::new(map.len() as u64);
                progress_bar.set_style(
                    ProgressStyle::default_bar()
                        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}").unwrap()
                        .progress_chars("##-"),
                );

                let hashed_tensors: Vec<Value> = map
                    .iter()
                    .filter(|(key, _)| key != &"__metadata__")
                    .collect::<Vec<_>>()
                    .par_iter()
                    .map(|(key, value)| {
                        progress_bar.inc(1);
                        progress_bar.set_message(format!("{}: {}", file_path.to_string(), key));
                        let hashed_tensor_results = hash_tensor(file_path, key, value);

                        hashed_tensor_results
                    })
                    .collect();
                
                progress_bar.finish_and_clear();

                for hashed_tensor in hashed_tensors {
                    let tensor_name = hashed_tensor["tensor"].as_str().unwrap().to_string();
                    output_object["tensors"][tensor_name] = hashed_tensor;
                }
            }
        }
        Err(e) => panic!("{}", format!("{}", e))
    }

    Ok(output_object)
}

fn hash_tensor(file_path: &str, tensor_name: &str, tensor_metadata: &Value) -> Value {
    // Re-read the file separately so that we can process each tensor separately
    let mut tensor_file = BufReader::new(File::open(file_path).unwrap());

    // Get the data_offsets
    let offsets = tensor_metadata.get("data_offsets").and_then(Value::as_array).unwrap();
    let offset_start = offsets[0].as_u64().unwrap();
    let offset_end = offsets[1].as_u64().unwrap();
    let offset_diff = offset_end - offset_start;
    // println!("Offset Difference: {:?}", offset_diff);

    // println!("Seeking to data");
    // Seek to the start position of the tensor data
    tensor_file.seek(SeekFrom::Start(offset_start)).unwrap();

    // Read tensor data into buffer
    // println!("Reading into buffer");
    let mut tensor_buffer = vec![0; (offset_diff / 8).try_into().unwrap()];
    tensor_file.read_exact(&mut tensor_buffer).unwrap();

    // Calculate SHA-256 hash of tensor data
    // println!("{} / {} Hashing...", index+1, map.len());
    let hash = sha256_hash(&tensor_buffer);
    // println!("SHA-256 Hash: {}", hash);
    let tensor_results = json!({
        "tensor": tensor_name,
        "hash": hash,
        "byte_count": offset_diff / 8,
        "offset_diff": offset_diff
    });

    tensor_results
}

fn sha256_hash(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let result = hasher.finalize();

    let hash_hex = format!("{:x}", result);

    hash_hex
}