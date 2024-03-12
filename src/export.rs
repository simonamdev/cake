use indicatif::{ProgressBar, ProgressStyle};
use serde_json::Value;
use std::fs::{self};
use std::io::prelude::*;
use std::path::PathBuf;

pub fn combine_cached_files_to_safetensors_file(
    model_id: &str,
    storage_directory: &str,
    target_file_path: &str,
) {
    println!("Exporting {} to {}...", model_id, target_file_path);

    let mut header_file_path = PathBuf::new();
    header_file_path.push(storage_directory);
    header_file_path.push(model_id);
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

    let header: Value = serde_json::from_slice(&header_bytes).unwrap();
    let tensor_names: Vec<String> = header
        .as_object()
        .unwrap()
        .keys()
        .filter(|key| *key != "__metadata__")
        .cloned()
        .collect();

    let tensor_count = tensor_names.len();

    let sty_main = ProgressStyle::with_template(
        "[{elapsed_precise}] {bar:40.green/yellow} {pos:>4}/{len:4} {msg}",
    )
    .unwrap();

    let main_bar: ProgressBar = ProgressBar::new(tensor_count as u64);
    main_bar.set_style(sty_main);

    for tensor_name in tensor_names {
        main_bar.set_message(format!("Writing {}", tensor_name));

        let mut tensor_file_path = PathBuf::new();
        tensor_file_path.push(storage_directory);
        tensor_file_path.push(model_id);
        tensor_file_path.push(tensor_name);
        let mut tensor_file = fs::File::open(tensor_file_path).unwrap();
        let mut tensor_bytes: Vec<_> = Vec::new();
        tensor_file.read_to_end(&mut tensor_bytes).unwrap();
        output_file.write_all(&tensor_bytes).unwrap();

        main_bar.inc(1);
    }

    main_bar.finish_with_message("Export complete");
}
