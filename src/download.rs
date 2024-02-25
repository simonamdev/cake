use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::iter::ParallelIterator;
use reqwest::{blocking::Client, Error};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::{self, metadata, File};
use std::io::prelude::*;
use std::path::PathBuf;
use std::time::Duration;

use rayon::prelude::*;

use crate::Layer;


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
    // Iterate through the tensors and write them to files

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

pub fn download_safetensors_file_by_model_id(model_id: &str) {
    // TODO: Support models with multiple files or files that aren't "model.safetensors"
    let url = &get_download_url_from_model_id(model_id, "model.safetensors");
    let cache_folder: &str = "./cache";
    download_safetensors_file(model_id, url, cache_folder);
    let target_file_path = "./test.safetensors";
    combine_cached_files_to_safetensors_file(cache_folder, target_file_path);
}

// Limitation: This currently does NOT make use of the hashing of layers to dedupe storage
// To do that, we would need to know the hashes in the model_id requested
// and to store the bytes of each tensor in a file with the hash as the name
// To do that, we could initially embed the hashes in the binary but that would be very large (200MB for 10% of HF textgen models)
// Ideally we download the hashes on demand from a "registry"
pub fn download_safetensors_file(model_id: &str, url: &str, storage_dir: &str) {
    // First download the header to understand the file
    let (header, header_length) = download_safetensors_header(url);

    // Write the header to a file
    // TODO: Move to its own function!
    let mut header_file_path = PathBuf::new();
    header_file_path.push(storage_dir);
    header_file_path.push(model_id);
    header_file_path.push("header.json");
    fs::create_dir_all(&header_file_path.parent().unwrap()).unwrap();
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
    // TODO! Move to its own function
    let mut header_length_path = PathBuf::new();
    header_length_path.push(storage_dir);
    header_length_path.push(model_id);
    header_length_path.push("header.length");
    fs::create_dir_all(&header_length_path.parent().unwrap()).unwrap();
    let mut header_length_file = File::create(header_length_path).unwrap();
    header_length_file
        .write_all(format!("{}", header_length).as_bytes())
        .unwrap();
    header_length_file.flush().unwrap();

    // Generate paths for every tensor
    let tensor_names_and_paths: Vec<(String, PathBuf)> = header
        .as_object()
        .unwrap()
        .iter()
        .filter_map(|data| {
            if data.0 != "__metadata__" {
                Some(data)
            } else {
                None
            }
        })
        .map(|(tensor_name, _)| {
            let mut tensor_file_path = PathBuf::new();
            tensor_file_path.push(storage_dir);
            tensor_file_path.push(model_id);
            tensor_file_path.push(tensor_name);
            (tensor_name.to_string(), tensor_file_path)
        })
        .filter_map(|data| {
            match !file_exists(&data.1) {
                true => Some(data),
                false => None,
            }
        }).collect();

    let tensor_names_and_paths_clone = tensor_names_and_paths.clone();
    // The way I've done it feels wasteful, but I'm sticking to it for now
    let tensor_names: Vec<String> = tensor_names_and_paths
        .into_iter()
        .map(|(name, _path)| {
            name
        })
        .collect();

    // Setup the progress bars
    let sty_main = ProgressStyle::with_template(
        "[{elapsed_precise}] {bar:40.green/yellow} {pos:>4}/{len:4}"
    )
        .unwrap();

    let main_bar = ProgressBar::new(header.as_object().unwrap().len() as u64);
    main_bar.set_style(sty_main);
    let main_bar_clone = main_bar.clone();
    let mp = MultiProgress::new();
    mp.add(main_bar);
    
    let layer_names_and_tensors: Vec<(Layer, Vec<u8>)> = par_download_layers(
        header, url.to_string(), Some(tensor_names), mp
    )
        .map(|data| {
            main_bar_clone.inc(1);
            data
        })
        .collect();

    let mut layer_name_to_tensor: HashMap<String, Vec<u8>> = HashMap::new();
    for (layer_name, tensor) in layer_names_and_tensors {
        layer_name_to_tensor.insert(layer_name.name, tensor);
    }

    // TODO: Do this in parallel as layers are downloaded instead of separately!
    for (name, file_path) in tensor_names_and_paths_clone {
        let tensor = layer_name_to_tensor.get(&name);
        match tensor {
            Some(t) => {
                // TODO: Add a progress spinenr to the multiprogress?
                // Write the tensor to the file path
                fs::create_dir_all(&file_path.parent().unwrap()).unwrap();
                let mut file = File::create(file_path). unwrap();
                file.write_all(t).unwrap();
                file.flush().unwrap();
            }
            None => {
                // Maybe display error message?
            }
        }
    }
}

pub fn par_download_layers(header: Value, url: String, tensor_names_allow_list: Option<Vec<String>>, mp: MultiProgress) -> impl ParallelIterator<Item = (Layer, Vec<u8>)> {
    // Setup spinners
    let sty_aux = ProgressStyle::with_template(
        "[{elapsed_precise}] {bar:20.cyan/blue} {pos:>8}/{len:8}B {msg}",
    )
        .unwrap()
        .progress_chars("##-");
    
    // Iterate over each tensor and download it
    // Only download the layers in the allow list, if it is available
    let layers: Vec<Layer> = header
        .as_object()
        .unwrap()
        .iter()
        // Skip the metadata key
        .filter_map(|data| {
            if data.0 != "__metadata__" {
                Some(data)
            } else {
                None
            }
        })
        // Skip tensors not in the allow list
        .filter_map(|data| {
            if tensor_names_allow_list.is_some() {
                if tensor_names_allow_list.as_ref().unwrap().contains(data.0) {
                    Some(data)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .map(|(name, metadata)| {
            let offsets = metadata.get("data_offsets").and_then(Value::as_array).unwrap();
            let offset_start = offsets[0].as_u64().unwrap();
            let offset_end = offsets[1].as_u64().unwrap();
            let offset_diff = offset_end - offset_start;
            Layer{
                name: name.to_string(),
                offset_start: offset_start,
                offset_end: offset_end,
                size: offset_diff
            }
        })
        .collect();

    let mut sorted_layers = layers.clone();
    sorted_layers.sort_by(|a: &Layer, b: &Layer| {
        b.size.cmp(&a.size)
    });

    sorted_layers
        .into_par_iter()
        .map(move |layer| {
            let pb = mp.add(ProgressBar::new(layer.size));
            pb.set_style(sty_aux.clone());
            pb.enable_steady_tick(Duration::from_millis(200));
            pb.set_message(format!("{}", layer.name));

            // Download the tensor
            // println!("{}: Downloading {}...", model_id, tensor_name);
            let tensor: Vec<u8> = download_tensor(
                &url,
                layer.offset_start,
                layer.offset_end,
                Some(pb.clone())
            ).unwrap(); // Handle unwrap better
            pb.finish_and_clear();
            (layer.clone(), tensor)
        })
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

fn download_tensor(url: &str, offset_start: u64, offset_end: u64, pb: Option<ProgressBar>) -> Result<Vec<u8>, Error> {
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
