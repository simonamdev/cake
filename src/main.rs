use std::fs::File;
use std::io::{self, BufReader, Read, Seek};

use serde_json::Value;
use sha2::{Sha256, Sha512, Digest};

fn main() {
    println!("Hello, world!");
    let file_path = "/home/simon/Downloads/models/mistral-7B-v0.1/model-00001-of-00002.safetensors";
    hash_safetensors_file(file_path).unwrap();
}

fn hash_safetensors_file(file_path: &str) -> io::Result<()> {
    println!("Hashing file: {}", file_path);
    // First 8 bytes represent the header
    let mut buffer = [0; 8];
    let mut file_for_header = File::open(file_path)?;
    file_for_header.read_exact(&mut buffer)?;

    // Convert to u64
    let json_header_length = u64::from_le_bytes(buffer);

    println!("{}", json_header_length);

    // Read the JSON header into a buffer
    let mut json_buffer = Vec::with_capacity(json_header_length.try_into().unwrap());
    let mut file_for_json = File::open(file_path)?;
    file_for_json.seek(io::SeekFrom::Start(8))?;
    file_for_json.take(json_header_length)
        .read_to_end(&mut json_buffer)?;

    // Switch the buffer to a string
    let json_string = String::from_utf8_lossy(&json_buffer);
    println!("{}", json_string);

    let json_value: Result<Value, _> = serde_json::from_str(&json_string);

    match json_value {
        Ok(value) => {
            if let Value::Object(map) = value {
                for (key, value) in map.iter() {
                    println!("Key: {}, Value: {}", key, value);
                    if key != "__metadata__" {
                        // Get the data_offsets
                        let offsets = value.get("data_offsets").and_then(Value::as_array).unwrap();
                        let offset_start = offsets[0].as_u64().unwrap();
                        let offset_end = offsets[1].as_u64().unwrap();
                        let offset_diff = offset_end - offset_start;
                        println!("{:?}", offset_diff);

                        let mut file_for_tensor = File::open(file_path)?;
                        file_for_tensor.seek(io::SeekFrom::Start(offset_start))?;

                        // The offsets determine the indexes at which the bytes exist
                        // In my understanding of the spec, offset DIFF is not number of BYTES, but number of BITS
                        // Therefore, to fill a buffer of BYTES, we should divide the diff by 8 to get each byte into each index
                        // TODO: Confirm the logic before proceeding with the analysis
                        let mut tensor_buffer: Vec<u8> = vec![0; (offset_diff / 8).try_into().unwrap()];
                        file_for_tensor.read_exact(&mut tensor_buffer)?;
                        // print!("{:?}", tensor_buffer);
                        println!("{:?}", tensor_buffer.last());
                        let hash = sha256_hash(&tensor_buffer);
                        println!("{}", hash);
                    }
                }
            } else {
                eprintln!("JSON is not an object");
            }
        }
        Err(e) => eprintln!("Error deserializing JSON: {}", e),
    }

    Ok(())
}

fn sha256_hash(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let result = hasher.finalize();
    
    let hash_hex = format!("{:x}", result);

    hash_hex
}