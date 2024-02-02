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
    let value = u64::from_le_bytes(buffer);

    println!("{}", value);

    Ok(())
}