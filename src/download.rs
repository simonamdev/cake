use serde_json::{json, Value};
use reqwest::{blocking::Client, Error};

pub fn download_safetensors_header(url: &str) -> (serde_json::Value) {
    // Step 1: download the first 8 bytes of the file, that contains the header length as u64

    let header_length_bytes: Vec<u8> = download_part_of_file(url, 0, 8).unwrap();
    println!("{:?}", header_length_bytes);
    let json_header_length = get_u64_from_u8_vec(header_length_bytes);

    println!("JSON header is {json_header_length} bytes long");

    let header_bytes: Vec<u8> = download_part_of_file(url, 8 - 1, json_header_length.try_into().unwrap()).unwrap();
    let json_string = String::from_utf8_lossy(&header_bytes);
    println!("{:}", json_string);
    println!("{}", json_header_length);
    println!("{}", json_string.len());
    println!("{}", header_bytes.len());
    let metadata_json: Result<Value, serde_json::Error> = serde_json::from_str(&json_string);

    match metadata_json {
        Ok(v) => {
            return json!({});
        }
        Err(e) => {
            eprintln!("{}", e);

            return json!({});
        }
    }

    // metadata_json
}

fn get_u64_from_u8_vec(mut bytes: Vec<u8>) -> u64 {
    let mut result: u64 = 0;
    // Rust does little endian by default, which means LSB comes first, but these bytes are big endian
    bytes.reverse();
    for &byte in &bytes {
        result = (result << 8) | u64::from(byte);
    }
    result
}

fn download_tensor(url: &str, tensor_length: usize) -> Vec<u8> {
    let mut tensor_buffer = vec![0; tensor_length];

    // TODO: download the tesnor using the Range header

    tensor_buffer
}

fn download_part_of_file(url: &str, byte_index: u64, number_of_bytes: u64) -> Result<Vec<u8>, Error> {

    // Range is exclusive. Example: 0-499 is byte 0 to byte 499, so 500 bytes in total
    let range_header_value = format!("bytes={}-{}", byte_index, byte_index + number_of_bytes - 1);
    let client = Client::new();
    let response = client.get(url)
        .header("Range", range_header_value)
        .send()?;

    let status_code = response.status();
    println!("Status Code: {}", status_code);

    let mut buffer: Vec<u8> = Vec::new();

    response.bytes().into_iter().for_each(|chunk| {
        buffer.extend_from_slice(&chunk);
    });

    return Ok(buffer);
}