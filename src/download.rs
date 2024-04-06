use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::iter::ParallelIterator;
use rayon::prelude::*;
use reqwest::{blocking::Client, Error};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::{self, metadata, File};
use std::io::prelude::*;
use std::io::Read;
use std::path::PathBuf;
use std::time::Duration;

use crate::{export, hf};
use crate::{hasher, Layer};

pub fn get_download_url_from_model_id(model_id: &str, file_name: &str) -> String {
    format!(
        "https://huggingface.co/{}/resolve/main/{}?download=true",
        model_id, file_name
    )
    .to_string()
}

pub fn download_safetensors_file_by_model_id(model_id: &str) {
    // Query the HF API to see the file names
    // TODO: Filter specifically for *.safetensors files
    let model_files_result = hf::get_model_files(model_id);
    if model_files_result.is_err() {
        // TODO: Handle better, print the error message too
        panic!("Unable to retrieve model files when attempting download!")
    }

    let model_files = model_files_result.unwrap();

    let safetensor_model_file_count = model_files
        .iter()
        .filter(|a| a.path.ends_with(".safetensors"))
        .count();

    // TODO: Support gguf files?
    if safetensor_model_file_count == 0 {
        // TODO: Handle with better error message
        panic!("No safetensors files found for the given model")
    }

    println!(
        "{} safetensors files to be downloaded",
        safetensor_model_file_count
    );

    // TODO: Configurable download folder, or pick a better sensible default
    let download_dir: &str = "./download";
    let mut file_index = 0;
    for file_info in model_files {
        println!(
            "File {} of {}: {}",
            file_index + 1,
            safetensor_model_file_count,
            file_info.path
        );

        let file_url = &get_download_url_from_model_id(model_id, &file_info.path);
        // TODO: Add the check here that will somehow find out if we already have some hashes on our filesystem
        // That depends on having a source of truth for the hashes
        // For now, a "hash registry" can be implemented
        // In future, maybe this field could be added to the safetensors spec itself

        // (NOT IMPLEMENTED YET) Retrieve the hashes for this model
        let (model_header, layers_to_hashes_map) =
            hasher::get_model_file_hashes(model_id, &file_info.path);

        // (NOT IMPLEMENTED YET) This check here is a placeholder for the future where we actually have the hashes available
        let locally_available_hashes =
            hasher::get_locally_available_hashes(layers_to_hashes_map.values().cloned().collect());

        let model_layers_to_download: Vec<_> = layers_to_hashes_map
            .iter()
            .filter(|&(_, v)| !locally_available_hashes.contains(v))
            .map(|(k, _)| k.to_string())
            .collect();

        // Setup the progress bars
        let main_bar = ProgressBar::new(model_layers_to_download.len() as u64).with_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:20.green/yellow} {pos:>4}/{len:4} {spinner:.blue} {msg}",
            )
            .unwrap(),
        );
        main_bar.enable_steady_tick(Duration::from_millis(500));
        let main_bar_clone = main_bar.clone();
        let mp: MultiProgress = MultiProgress::new();
        mp.add(main_bar);
        let layers_downloaded_count = model_layers_to_download.len();

        par_download_layers(
            model_header.raw_header,
            file_url.to_string(),
            Some(model_layers_to_download),
            mp,
        )
        .map(|(layer, layer_bytes)| {
            // Generate the hash
            // TODO: Once hashing is actually available from the registry, we should double check
            // that the hashes match
            let layer_hash = hasher::sha256_hash(&layer_bytes);

            (layer, layer_hash, layer_bytes)
        })
        .for_each(|(layer, layer_hash, layer_bytes)| {
            // Decide where to store the this layer by its hash
            // TODO: Handle this failure case ahead of time somehow
            let mut hashed_layer_path = PathBuf::new();
            hashed_layer_path.push(download_dir);
            hashed_layer_path.push(layer_hash);

            // Create the directory if its does not exist
            fs::create_dir_all(hashed_layer_path.parent().unwrap()).unwrap();
            // Write the file
            let mut file = File::create(hashed_layer_path).unwrap();
            file.write_all(&layer_bytes).unwrap();
            file.flush().unwrap();
            // Increment the progress bar
            main_bar_clone.inc(1);
        });

        main_bar_clone.finish_with_message(format!(
            "{} All done! {} layers already present, {} layers downloaded",
            file_info.path,
            layers_to_hashes_map.len() - layers_downloaded_count,
            layers_downloaded_count
        ));

        file_index += 1;
    }

    // TODO: Add support to export the safetensors file/s conditionally at the end
}

// Limitation: This currently does NOT make use of the hashing of layers to dedupe storage
// To do that, we would need to know the hashes in the model_id requested
// and to store the bytes of each tensor in a file with the hash as the name
// To do that, we could initially embed the hashes in the binary but that would be very large (200MB for 10% of HF textgen models)
// Ideally we download the hashes on demand from a "registry"
pub fn download_safetensors_file(model_id: &str, file_url: &str, storage_dir: &str) {
    // First download the header to understand the file
    println!("Retrieving header for {}", model_id);
    let (header, header_length) = download_safetensors_header(file_url);
    // TODO: Handle when header_length is zero

    // Write the header to a file
    // TODO: Move to its own function!
    let mut header_file_path = PathBuf::new();
    header_file_path.push(storage_dir);
    header_file_path.push(model_id);
    header_file_path.push("header.json");
    fs::create_dir_all(header_file_path.parent().unwrap()).unwrap();
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
    fs::create_dir_all(header_length_path.parent().unwrap()).unwrap();
    let mut header_length_file = File::create(header_length_path).unwrap();
    header_length_file
        .write_all(format!("{}", header_length).as_bytes())
        .unwrap();
    header_length_file.flush().unwrap();

    // Generate paths for every tensor
    // We will use this to not download tensors that are already downloaded
    let mut tensor_name_to_path: HashMap<String, PathBuf> = HashMap::new();
    header
        .as_object()
        .unwrap()
        .iter()
        .filter(|data| data.0 != "__metadata__")
        .map(|(tensor_name, _)| {
            let mut hashed_layer_path = PathBuf::new();
            hashed_layer_path.push(storage_dir);
            hashed_layer_path.push(model_id);
            hashed_layer_path.push(tensor_name);
            (tensor_name.to_string(), hashed_layer_path)
        })
        .filter(|data| {
            // We only want the ones where the file does NOT exist yet
            !file_exists(&data.1)
        })
        .for_each(|(name, path)| {
            tensor_name_to_path.insert(name, path);
        });

    let tensor_names_for_download: Vec<String> = tensor_name_to_path.keys().cloned().collect();

    // Setup the progress bars
    // TODO: Dedupe this
    let sty_main =
        ProgressStyle::with_template("[{elapsed_precise}] {bar:40.green/yellow} {pos:>4}/{len:4}")
            .unwrap();

    let main_bar: ProgressBar = ProgressBar::new(tensor_names_for_download.len() as u64);
    main_bar.set_style(sty_main);
    let main_bar_clone = main_bar.clone();
    let mp = MultiProgress::new();
    mp.add(main_bar);

    par_download_layers(
        header,
        file_url.to_string(),
        Some(tensor_names_for_download),
        mp,
    )
    .for_each(|(layer, tensor)| {
        // Write to file
        let file_path = tensor_name_to_path.get(&layer.name).unwrap();
        fs::create_dir_all(file_path.parent().unwrap()).unwrap();
        let mut file = File::create(file_path).unwrap();
        file.write_all(&tensor).unwrap();
        file.flush().unwrap();
        // Increment the progress bar
        // TODO: also add a new spinner indicating writing to file later
        main_bar_clone.inc(1);
    });
}

pub fn par_download_layers(
    header: Value,
    file_url: String,
    tensor_names_allow_list: Option<Vec<String>>,
    mp: MultiProgress,
) -> impl ParallelIterator<Item = (Layer, Vec<u8>)> {
    // print!("{}", header);

    // Setup the reqwest client to enable connection pooling
    let client = Client::new();

    // Iterate over each tensor and download it
    // Only download the layers in the allow list, if it is available
    let layers: Vec<Layer> = header
        .as_object()
        .unwrap()
        .iter()
        // Skip the metadata key
        .filter(|data| data.0 != "__metadata__")
        // Skip tensors not in the allow list
        .filter(|data| {
            if let Some(tnal) = &tensor_names_allow_list {
                tnal.contains(data.0)
            } else {
                true
            }
        })
        .map(|(name, metadata)| {
            let offsets = metadata
                .get("data_offsets")
                .and_then(Value::as_array)
                .unwrap();
            let offset_start = offsets[0].as_u64().unwrap();
            let offset_end = offsets[1].as_u64().unwrap();
            let offset_diff = offset_end - offset_start;
            Layer {
                name: name.to_string(),
                offset_start,
                offset_end,
                size: offset_diff,
            }
        })
        .collect();
    // println!("{:?}", layers);

    let mut sorted_layers = layers.clone();
    sorted_layers.sort_by(|a: &Layer, b: &Layer| b.size.cmp(&a.size));

    // println!("{:?}", sorted_layers);

    sorted_layers.into_par_iter().map(move |layer| {
        let client = &client;
        let pb = mp.add(ProgressBar::new(layer.size));
        pb.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:20.cyan/blue} {pos:>8}/{len:8}B {msg}",
            )
            .unwrap(),
        );
        pb.enable_steady_tick(Duration::from_millis(200));
        pb.set_message(layer.name.to_string());

        // Download the tensor
        // println!("{}: Downloading {}...", model_id, tensor_name);
        let tensor: Vec<u8> = download_tensor(
            &file_url,
            layer.offset_start,
            layer.offset_end,
            client,
            Some(pb.clone()),
        )
        .unwrap(); // Handle unwrap better
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

pub fn download_safetensors_header(file_url: &str) -> (serde_json::Value, u64) {
    let client = Client::new();

    // Step 1: download the first 8 bytes of the file, that contains the header length as u64

    let header_length_bytes: Vec<u8> =
        download_part_of_file(file_url, 0, 8, &client, None).unwrap();
    let ret = String::from_utf8_lossy(&header_length_bytes).to_string();
    println!("{}", ret);

    // println!("{:?}", header_length_bytes);
    let json_header_length = get_u64_from_u8_vec(header_length_bytes);

    match json_header_length {
        Some(jhl) => {
            println!("JSON header is {jhl} bytes long");

            let header_bytes: Vec<u8> =
                download_part_of_file(file_url, 8, jhl, &client, None).unwrap();
            let json_string = String::from_utf8_lossy(&header_bytes);
            // println!("{:}", json_string);
            // println!("{}", json_header_length);
            // println!("{}", json_string.len());
            // println!("{}", header_bytes.len());
            let metadata_json: serde_json::Value = serde_json::from_str(&json_string).unwrap();

            (metadata_json, jhl)
        }
        None => (json!({}), 0),
    }
}

fn get_u64_from_u8_vec(bytes: Vec<u8>) -> Option<u64> {
    let b = bytes.try_into();
    match b {
        Ok(bytes) => {
            println!("{:?}", bytes);
            Some(u64::from_le_bytes(bytes))
        }
        Err(_e) => {
            // println!("{:?}", e);
            None
        }
    }
}

fn download_tensor(
    file_url: &str,
    offset_start: u64,
    offset_end: u64,
    client: &Client,
    pb: Option<ProgressBar>,
) -> Result<Vec<u8>, Error> {
    let offset_diff = offset_end - offset_start;
    let byte_count = offset_diff;

    download_part_of_file(file_url, offset_start, byte_count, client, pb)
}

fn download_part_of_file(
    file_url: &str,
    byte_index: u64,
    number_of_bytes: u64,
    client: &Client,
    pb: Option<ProgressBar>,
) -> Result<Vec<u8>, Error> {
    let chunk_size = 1024; // 1KB

    // Range is exclusive. Example: 0-499 is byte 0 to byte 499, so 500 bytes in total
    let range_header_value = format!("bytes={}-{}", byte_index, byte_index + number_of_bytes - 1);
    let mut response = client
        .get(file_url)
        .header("Range", range_header_value)
        .send()?;

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

    Ok(buffer)
}
