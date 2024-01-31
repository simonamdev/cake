fn main() {
    println!("Hello, world!");
    let file_path = "test";
    hash_safetensors_file(file_path.to_string());
}

fn hash_safetensors_file(file_path: String) {
    println!("Hashing file: {file_path}");
}