use std::env;
use std::fs::{self, File};
use std::io::Read;
use std::convert::TryInto;

use clap::{Parser, Subcommand};

use serde_json::{json, Map, Value};

use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};

mod download;
mod hash;
mod compare;

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

    Download {
        #[arg(long)]
        model_id: String,
    }
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
            let url = &download::get_download_url_from_model_id(model_id, "model.safetensors");
            let download_folder = "./download";
            let cache_folder: &str = "./cache";
            download::download_full_safetensors_file(url, download_folder, cache_folder);
            let target_file_path = "./test.safetensors";
            download::combine_cached_files_to_safetensors_file(cache_folder, target_file_path);
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
    let mut i = 0;
    for (model_id, file_names) in json.as_object().unwrap() {
        i += 1;
        println!("{}/{}: {}", i, json.as_object().unwrap().len(), model_id);
        // For now, handle only models with a single file, for simplicity
        // TODO: Handle multi file models!
        if file_names.as_array().unwrap().len() > 1 {
            println!("{} skipped due to multiple safetensors files. (temporary limitation)", model_id);
            continue
        }
        if file_names.as_array().unwrap().len() == 0 {
            println!("{} skipped due to no files", model_id);
            continue
        }
        let safetensors_file_name = file_names.as_array().unwrap().get(0).unwrap().as_str().unwrap();
        // For now skip adapter models as they are not being parsed correctly
        if safetensors_file_name.contains("adapter_model") {
            println!("{} skipped due to being an adapter model file (temporary limitation)", model_id);
            continue;
        }
        let model_parts: Vec<&str> = model_id.split("/").collect();
        let hashes_file_path = get_hashes_file_dir_and_path(
            model_parts[0],
            model_parts[1]
        );
        let hashes_file_path_clone = hashes_file_path.1.clone();
        let hashes_file_exists = fs::metadata(hashes_file_path_clone).is_ok();
        if hashes_file_exists {
            println!("{} skipped as hashes file already exists", model_id);
            continue
        }
        let hashed_layers_result = download_and_hash_layers(model_id, safetensors_file_name);
        fs::create_dir_all(hashes_file_path.0).unwrap();
        let file = File::create(hashes_file_path.1).unwrap();
        serde_json::to_writer_pretty(file, &hashed_layers_result).unwrap();
    }


    // Download layers and hash each one
    let model_id = "KoboldAI/fairseq-dense-1.3B";
    let file_name = "model.safetensors";

    let hashed_layers_result = download_and_hash_layers(model_id, file_name);
    println!("{:#?}", hashed_layers_result);
}

fn download_and_hash_layers(model_id: &str, file_name: &str) -> Map<String, Value> {
    let url = &download::get_download_url_from_model_id(model_id, file_name);

    let (header, _) = download::download_safetensors_header(url);

    // Iterate over each tensor, download it and hash the layer
    // Convert the JSON object into a slice of mutable key-value pairs
    let header_entries: Vec<(&String, &Value)> = header.as_object().unwrap().iter().collect();

    let bar = ProgressBar::new(header_entries.len().try_into().unwrap());
    bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {percent}% {msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    // Process the header entries in parallel
    let processed_entries: Vec<(String, Value)> = header_entries
        .par_iter()
        .filter_map(|(tensor_name, tensor_metadata)| {
            if *tensor_name == "__metadata__" {
                bar.inc(1);
                None
            } else {
                // TODO: dedupe this
                let offsets = tensor_metadata.get("data_offsets").and_then(Value::as_array)?;
                let offset_start = offsets[0].as_u64()?;
                let offset_end = offsets[1].as_u64()?;
                // Download the tensor
                println!("{}: Downloading {}...", model_id, tensor_name);
                let tensor = download::download_tensor(url, offset_start, offset_end).unwrap(); // Handle unwrap better
                // Hash the tensor
                println!("Hashing {}...", tensor_name);
                let hash = hash::sha256_hash(&tensor);
                // Put that in the results
                let tensor_result = json!({
                    "data_offsets": offsets,
                    "hash": hash
                });
                bar.inc(1);
                Some((tensor_name.to_string(), tensor_result))
            }
        })
        .collect();

    bar.finish();

    // Create a new map and insert processed entries
    let mut result_obj: Map<String, Value> = Map::new();
    for (tensor_name, tensor_result) in processed_entries {
        result_obj.insert(tensor_name, tensor_result);
    }


    result_obj
}

fn get_hashes_file_dir_and_path(model_account: &str, model_name: &str) -> (String, String) {
    let abs_dir: String =  env::current_dir().unwrap().to_string_lossy().into_owned().clone() + &"/results/".to_owned() + &model_account + "/" + &model_name;
    let target_file_path = abs_dir.to_owned() + "/hashes.json";

    (abs_dir, target_file_path)
}
