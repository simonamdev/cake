use std::fs::File;
use std::io::{self, Read};

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
    file.take(json_header_length).read_to_end(&mut json_buffer)?;

    println!("{:?}", json_buffer);

    Ok(())
}