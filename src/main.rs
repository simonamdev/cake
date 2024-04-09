use std::env;
use std::fs::{self, File};
use std::io::Read;
use std::time::Duration;

use clap::{Parser, Subcommand};

use rayon::iter::ParallelIterator;
use serde_json::{json, Map, Value};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use lz4::block::compress;

mod compare;
mod download;
mod export;
mod hasher;
mod hf;
mod registry;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    HashingExperiment {},

    Compare {
        #[arg(long)]
        a: String,
        #[arg(long)]
        b: String,
    },

    CheckModels {},

    Download {
        #[arg(long)]
        model_id: String,
    },

    Registry {},
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::HashingExperiment {}) => {
            run_hashing_experiment();
        }
        Some(Commands::Compare { a, b }) => {
            compare::compare_tensors_between_files(a, b);
        }
        Some(Commands::Download { model_id }) => {
            // Download safetensor files in pieces then create a new safetensors files
            // Known issue: using this will not create an equivalent file to that available on huggingface due to
            // differences in how the json header is formatted, however it will create a valid safetensors file
            // Known issue: only works for models with a single file called "model.safetensors"
            download::download_safetensors_file_by_model_id(model_id)
        }
        Some(Commands::CheckModels {}) => {
            // Get the model ids and file names from the JSON file
            let mut file = File::open("safetensor-models-text-gen.json").unwrap();
            let mut models_json_str = String::new();
            file.read_to_string(&mut models_json_str).unwrap();

            let json: Value = serde_json::from_str(&models_json_str).unwrap();
            let mut i = 0;
            for (model_id, file_names) in json.as_object().unwrap() {
                i += 1;
                if i < 1727 {
                    // Update this as we fix issues
                    continue;
                }
                for file_name in file_names.as_array().unwrap() {
                    println!(
                        "{}/{}: {} ({})",
                        i,
                        json.as_object().unwrap().len(),
                        model_id,
                        file_name
                    );
                    let model_parts: Vec<&str> = model_id.split('/').collect();
                    let hashes_file_path =
                        get_hashes_file_dir_and_path(model_parts[0], model_parts[1]);
                    let hashes_file_path_clone = hashes_file_path.1.clone();
                    let hashes_file_exists = fs::metadata(hashes_file_path_clone).is_ok();
                    if hashes_file_exists {
                        println!("{} skipped as hashes file already exists", model_id);
                        continue;
                    }
                    // Download just the header and try to parse it
                    let url = &download::get_download_url_from_model_id(
                        model_id,
                        file_name.as_str().unwrap(),
                    );
                    download::download_safetensors_header(url);
                }
            }
        }
        Some(Commands::Registry {}) => {
            registry::run_registry();
        }
        None => {}
    }
}

fn run_hashing_experiment() {
    // Get the model ids and file names from the JSON file
    let mut file = File::open("safetensor-models-text-gen.json").unwrap();
    let mut models_json_str = String::new();
    file.read_to_string(&mut models_json_str).unwrap();

    let json: Value = serde_json::from_str(&models_json_str).unwrap();
    let mut model_index = 0;
    for (model_id, file_names) in json.as_object().unwrap() {
        let mut file_index = 1;
        model_index += 1;
        println!(
            "{}/{}: {}.",
            model_index,
            json.as_object().unwrap().len(),
            model_id,
        );
        if file_names.as_array().unwrap().is_empty() {
            println!("{} skipped due to no files", model_id);
            continue;
        }

        let model_parts: Vec<&str> = model_id.split('/').collect();
        let hashes_file_path = get_hashes_file_dir_and_path(model_parts[0], model_parts[1]);
        let hashes_file_path_clone = hashes_file_path.1.clone();
        let hashes_file_exists = fs::metadata(hashes_file_path_clone).is_ok();
        if hashes_file_exists {
            println!("{} skipped as hashes file already exists", model_id);
            continue;
        }

        // Download each file separately and then merge the results if there are multiple files
        let mut file_results: Vec<Map<String, Value>> = Vec::new();
        for file_name_val in file_names.as_array().unwrap() {
            let file_name = file_name_val.as_str().unwrap().to_string();
            // For now skip adapter models as they are not being parsed correctly
            if file_name.contains("adapter_model") {
                println!(
                    "{} skipped due to being an adapter model file (temporary limitation)",
                    model_id
                );
                continue;
            }

            println!(
                "[File {}/{}] Downloading model layers from {}",
                file_index,
                file_names.as_array().unwrap().len(),
                file_name,
            );
            let hashed_layers_result = download_and_hash_layers(model_id, &file_name);
            // If no results are returned, skip this file
            if hashed_layers_result.is_empty() {
                println!("{} skipped due to invalid header length", model_id);
                continue;
            }
            file_results.push(hashed_layers_result);
            file_index += 1;
        }

        // Merge the results together if there are multiple
        let mut output_result: Map<String, Value> = Map::new();
        if file_results.len() == 1 {
            // TODO: Not sure if there's an idiomatic way to avoid this clone
            output_result = file_results.first().unwrap().clone()
        } else {
            for result in file_results.iter() {
                for (key, value) in result {
                    // TODO: Another clone which feels like it can be avoided
                    output_result.insert(key.to_string(), value.clone());
                }
            }
        }

        fs::create_dir_all(hashes_file_path.0).unwrap();
        let file = File::create(hashes_file_path.1).unwrap();
        println!("Outputting hash results...");
        serde_json::to_writer_pretty(file, &output_result).unwrap();
    }
}

#[derive(Debug, Eq, Ord, PartialEq, PartialOrd, Clone)]
struct Layer {
    name: String,
    offset_start: u64,
    offset_end: u64,
    size: u64,
}

struct LayerMetadata {
    layer: Layer,
    hash: String,
    size: u64,
    compressed_hash: String,
    compressed_size: i64,
}

fn download_and_hash_layers(model_id: &str, file_name: &str) -> Map<String, Value> {
    // Create a new map and insert processed entries
    let mut result_obj: Map<String, Value> = Map::new();

    // Get the header of the model
    let url = download::get_download_url_from_model_id(model_id, file_name);
    let (header, header_length) = download::download_safetensors_header(&url);
    if header_length == 0 {
        println!("No header returned!");
        println!("{}", header);
        return result_obj;
    }

    // Setup the progress bars
    let main_bar = ProgressBar::new(header.as_object().unwrap().len() as u64).with_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:20.green/yellow} {pos:>4}/{len:4} {spinner:.blue} {msg}",
        )
        .unwrap(),
    );
    main_bar.enable_steady_tick(Duration::from_millis(500));
    let main_bar_clone = main_bar.clone();
    let mp: MultiProgress = MultiProgress::new();
    mp.add(main_bar);

    let layers_metadata: Vec<LayerMetadata> = download::par_download_layers(header, url, None, mp)
        .map(|(layer, tensor)| {
            // Perform the hashing part for uncompressed version
            main_bar_clone.set_message(format!("Hashing: {}", layer.name));
            let hash = hasher::sha256_hash(&tensor);
            // Perform the hashing part for compressed version
            // Compress the tensor
            main_bar_clone.set_message(format!("Compressing: {}", layer.name));
            let compressed_tensor = compress(&tensor, None, false);
            main_bar_clone.set_message(format!("Hashing Compressed: {}", layer.name));
            let mut compressed_hash: String = "N/A".to_string();
            let mut compressed_size: i64 = -1;
            match compressed_tensor {
                Ok(ct) => {
                    compressed_hash = hasher::sha256_hash(&ct);
                    compressed_size = ct.len() as i64;
                }
                Err(_e) => {}
            }
            main_bar_clone.inc(1);
            main_bar_clone.set_message("Waiting...");

            LayerMetadata {
                layer,
                hash,
                size: tensor.len() as u64,
                compressed_hash,
                compressed_size,
            }
        })
        .collect();

    main_bar_clone.finish_with_message("All done!");

    for layer_metadata in layers_metadata {
        let tensor_result = json!({
            "data_offsets": vec![layer_metadata.layer.offset_start, layer_metadata.layer.offset_end],
            "hash": layer_metadata.hash,
            "compressed_hash": layer_metadata.compressed_hash,
            "size": layer_metadata.size,
            "compressed_size": layer_metadata.compressed_size,
            "file_name": file_name,
        });
        result_obj.insert(layer_metadata.layer.name, tensor_result);
    }

    result_obj
}

fn get_hashes_file_dir_and_path(model_account: &str, model_name: &str) -> (String, String) {
    let abs_dir: String = env::current_dir()
        .unwrap()
        .to_string_lossy()
        .into_owned()
        .clone()
        + "/results/"
        + model_account
        + "/"
        + model_name;
    let target_file_path = abs_dir.to_owned() + "/hashes.json";

    (abs_dir, target_file_path)
}
