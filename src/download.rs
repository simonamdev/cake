use indicatif::ProgressBar;
use reqwest::{blocking::Client, Error};
use serde_json::Value;
use std::fs::{self, metadata, File};
use std::io::prelude::*;
use std::path::PathBuf;


pub fn get_download_url_from_model_id(model_id: &str, file_name: &str) -> String {
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}?download=true",
        model_id, file_name
    )
    .to_string();
    return url;
}

pub fn combine_cached_files_to_safetensors_file(cache_directory: &str, target_file_path: &str) {
    let mut header_file_path = PathBuf::new();
    header_file_path.push(cache_directory);
    header_file_path.push("header.json");

    let mut header_file = fs::File::open(header_file_path).unwrap();
    let mut header_bytes = Vec::new();
    header_file.read_to_end(&mut header_bytes).unwrap();

    // println!("{}", header_bytes.len());

    let mut output_file = fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(target_file_path)
        .unwrap();
    // Write the header length
    let header_length_bytes = &header_bytes.len().to_le_bytes();
    println!("{:?}", header_length_bytes);
    output_file.write_all(header_length_bytes).unwrap();
    println!("{:?}", header_bytes.len());
    // Write the header bytes
    output_file.write_all(&header_bytes).unwrap();
    // Iterate through the tensors and write them

    let mut header: Value = serde_json::from_slice(&header_bytes).unwrap();
    if let Some(obj) = header.as_object_mut() {
        // Sort the JSON object by the first index of the "data_offsets" array
        if let Some(data_offsets) = obj.get_mut("data_offsets") {
            if let Some(data_offsets_array) = data_offsets.as_array_mut() {
                data_offsets_array.sort_by(|a, b| {
                    let a_first = a[0].as_i64().unwrap_or(i64::MAX);
                    let b_first = b[0].as_i64().unwrap_or(i64::MAX);
                    a_first.cmp(&b_first)
                });
            }
        }

        for (key, value) in obj {
            println!("Key: {}, Value: {}", key, value);
            if key == "__metadata__" {
                continue;
            }

            let mut tensor_file_cache_path = PathBuf::new();
            tensor_file_cache_path.push(cache_directory);
            tensor_file_cache_path.push(key);
            let mut tensor_file = fs::File::open(tensor_file_cache_path).unwrap();
            let mut tensor_bytes: Vec<_> = Vec::new();
            tensor_file.read_to_end(&mut tensor_bytes).unwrap();
            output_file.write_all(&tensor_bytes).unwrap();
        }
    }
}

pub fn download_full_safetensors_file(url: &str, _download_directory: &str, cache_directory: &str) {
    // First download the header to understand the file
    let (header, header_length) = download_safetensors_header(url);

    // Write the header to a file
    let mut header_file_path = PathBuf::new();
    header_file_path.push(cache_directory);
    header_file_path.push("header.json");
    let mut header_file = File::create(header_file_path).unwrap();
    let mut header_buf = serde_json::to_string(&header).unwrap().into_bytes();
    // TAKEN DIRECTLY FROM SAFETENSORS -|
    // Force alignment to 8 bytes.
    let extra = (8 - header_buf.len() % 8) % 8;
    header_buf.extend(vec![b' '; extra]);
    // TAKEN DIRECTLY FROM SAFETENSORS -|
    header_file.write_all(&header_buf).unwrap();
    header_file.flush().unwrap();

    // Write the header length to a file
    let mut header_length_path = PathBuf::new();
    header_length_path.push(cache_directory);
    header_length_path.push("header.length");
    let mut header_length_file = File::create(header_length_path).unwrap();
    header_length_file
        .write_all(format!("{}", header_length).as_bytes())
        .unwrap();
    header_length_file.flush().unwrap();

    for (key, value) in header.as_object().unwrap() {
        if key == "__metadata__" {
            continue;
        }
        // Here the key is the tensor name and the value is in the format:
        // {"data_offsets":[1209081856,1217470464],"dtype":"F16","shape":[2048,2048]}
        // println!("{} {}", key, value);
        // If the file exists, skip it
        let mut tensor_file_cache_path = PathBuf::new();
        tensor_file_cache_path.push(cache_directory);
        tensor_file_cache_path.push(key);
        if file_exists(&tensor_file_cache_path) {
            println!(
                "{} already exists, skipping...",
                tensor_file_cache_path.display()
            );
            continue;
        }
        let offsets = value.get("data_offsets").and_then(Value::as_array).unwrap();
        let offset_start = offsets[0].as_u64().unwrap();
        let offset_end = offsets[1].as_u64().unwrap();
        // Download the tensor
        let tensor = download_tensor(url, offset_start, offset_end, None).unwrap();
        // Write the tensor to the cache dir file
        let mut file = File::create(&tensor_file_cache_path).unwrap();
        println!("Writing {} to {}...", key, tensor_file_cache_path.display());
        file.write_all(&tensor).unwrap();
        file.flush().unwrap();
    }
}

fn file_exists(path: &PathBuf) -> bool {
    if let Ok(metadata) = metadata(path) {
        metadata.is_file()
    } else {
        false
    }
}

pub fn download_safetensors_header(url: &str) -> (serde_json::Value, u64) {
    // Step 1: download the first 8 bytes of the file, that contains the header length as u64

    let header_length_bytes: Vec<u8> = download_part_of_file(url, 0, 8, None).unwrap();
    println!("{:?}", header_length_bytes);
    let json_header_length = get_u64_from_u8_vec(header_length_bytes);

    println!("JSON header is {json_header_length} bytes long");

    let header_bytes: Vec<u8> =
        download_part_of_file(url, 8, json_header_length.try_into().unwrap(), None).unwrap();
    let json_string = String::from_utf8_lossy(&header_bytes);
    // println!("{:}", json_string);
    // println!("{}", json_header_length);
    // println!("{}", json_string.len());
    // println!("{}", header_bytes.len());
    let metadata_json: serde_json::Value = serde_json::from_str(&json_string).unwrap();

    (metadata_json, json_header_length)
}

fn get_u64_from_u8_vec(bytes: Vec<u8>) -> u64 {
    let b: [u8; 8] = bytes.try_into().unwrap();
    u64::from_le_bytes(b)
}

pub fn download_tensor(url: &str, offset_start: u64, offset_end: u64, pb: Option<ProgressBar>) -> Result<Vec<u8>, Error> {
    let offset_diff = offset_end - offset_start;
    let byte_count = offset_diff;

    let tensor = download_part_of_file(url, offset_start, byte_count, pb);

    tensor
}

fn download_part_of_file(
    url: &str,
    byte_index: u64,
    number_of_bytes: u64,
    pb: Option<ProgressBar>
) -> Result<Vec<u8>, Error> {
    let chunk_size = 1024; // 1KB
    
    // Range is exclusive. Example: 0-499 is byte 0 to byte 499, so 500 bytes in total
    let range_header_value = format!("bytes={}-{}", byte_index, byte_index + number_of_bytes - 1);
    let client = Client::new();
    let mut response = client.get(url).header("Range", range_header_value).send()?;

    // let status_code = response.status();
    // println!("Status Code: {}", status_code);

    // Setup the progress bar if one is available
    let total_size = response.content_length().unwrap_or(0);
    if let Some(pb) = &pb {
        pb.set_length(total_size);
    }

    let mut buffer: Vec<u8> = Vec::new();
    let mut chunk = vec![0; chunk_size];
    let mut position: usize = 0;

    while let Ok(chunk_size) = response.read(&mut chunk) {
        if chunk_size == 0 {
            break; // End of stream
        }
        
        position += chunk_size;
        buffer.extend_from_slice(&chunk[..chunk_size]);
        if let Some(pb) = &pb {
            pb.set_position(position as u64);
        }
    }

    return Ok(buffer);
}
