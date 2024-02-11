use std::fs::File;
use std::io::{self, BufReader, ErrorKind, Read, Seek, SeekFrom};
use std::convert::TryInto;

use serde_json::{json, value, Error, Value};
use sha2::{Sha256, Digest};

fn main() {
    let file_path = "/home/simon/Downloads/models/mistral-7B-v0.1/model-00001-of-00002.safetensors";
    let hashed_model_result = hash_safetensors_file(file_path);
    match hashed_model_result {
        Ok(value) => {
            println!("{}", value);
        }
        Err(e) => {
            eprintln!("{}", e);
        }
    }
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