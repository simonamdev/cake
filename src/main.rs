use std::fs::File;
use std::io::{self, Read};

use serde_json::Value;

fn main() {
    println!("Hello, world!");
    let file_path = "/home/simon/Downloads/models/mistral-7B-v0.1/model-00001-of-00002.safetensors";
    hash_safetensors_file(file_path.to_string()).unwrap();
}

fn hash_safetensors_file(file_path: String) -> io::Result<()> {
    println!("Hashing file: {file_path}");
    let mut file = File::open(file_path)?;

    // First 8 bytes represent the header
    let mut buffer = [0; 8];
    file.read_exact(&mut buffer)?;

    // convert to u64
    let json_header_length = u64::from_le_bytes(buffer);

    println!("{}", json_header_length);

    // This unwrap feels odd here but what do I know I'm new to rust
    let mut json_buffer = Vec::with_capacity(json_header_length.try_into().unwrap());
    file.take(json_header_length)
        .read_to_end(&mut json_buffer)?;

    // println!("{:?}", json_buffer);
    // Switch the buffer to a string
    let json_string = String::from_utf8_lossy(&json_buffer);

    let json_value: Result<Value, _> = serde_json::from_str(&json_string);

    match json_value {
        Ok(value) => {
            // Query keys
            if let Value::Object(map) = value {
                for (key, value) in map.iter() {
                    println!("Key: {}, Value: {}", key, value);
                    if key != "__metadata__" {
                        // Get the data_offsets
                        let offsets = value.get("data_offsets").and_then(Value::as_array).unwrap();
                        let offset_start = offsets[0].as_u64().unwrap();
                        let offset_end = offsets[1].as_u64().unwrap();
                        let offset_diff = offset_end - offset_start;
                        println!("{:?}", offset_diff)
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
