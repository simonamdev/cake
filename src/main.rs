use std::env;
use std::fs::{self, File};
use std::io::Read;
use std::time::Duration;

use clap::{Parser, Subcommand};

use serde_json::{json, Map, Value};

use rayon::prelude::*;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

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

    CheckModels {},

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
            // Known issue: only works for models with a single file called "model.safetensors"
            let url = &download::get_download_url_from_model_id(model_id, "model.safetensors");
            let cache_folder: &str = "./cache";
            download::download_full_safetensors_file(url, cache_folder);
            let target_file_path = "./test.safetensors";
            download::combine_cached_files_to_safetensors_file(cache_folder, target_file_path);
        }
        Some(Commands::CheckModels {  }) => {
            // Get the model ids and file names from the JSON file
            let mut file = File::open("safetensor-models-text-gen.json").unwrap();
            let mut models_json_str = String::new();
            file.read_to_string(&mut models_json_str).unwrap();

            let json: Value = serde_json::from_str(&models_json_str).unwrap();
            let mut i = 0;
            for (model_id, file_names) in json.as_object().unwrap() {
                i += 1;
                if i < 205 {
                    // Update this as we fix issues
                    continue;
                }
                for file_name in file_names.as_array().unwrap() {
                    println!("{}/{}: {} ({})", i, json.as_object().unwrap().len(), model_id, file_name);
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
                    // Download just the header and try to parse it
                    let url = &download::get_download_url_from_model_id(model_id, file_name.as_str().unwrap());
                    download::download_safetensors_header(url);
                }
            }
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

#[derive(Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Layer {
    name: String,
    offset_start: u64,
    offset_end: u64,
    size: u64
}

fn download_and_hash_layers(model_id: &str, file_name: &str) -> Map<String, Value> {
    let url = &download::get_download_url_from_model_id(model_id, file_name);

    let (header, _) = download::download_safetensors_header(url);

    // Iterate over each tensor, download it and hash the layer
    // Convert the JSON object into a slice of mutable key-value pairs
    let header_entries: Vec<(&String, &Value)> = header.as_object().unwrap().iter().collect();
    let mut layers: Vec<Layer> = header
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

    layers.sort_by(|a: &Layer, b: &Layer| {
        b.size.cmp(&a.size)
    });

    println!("{:?}", layers);

    let sty_main = ProgressStyle::with_template("[{elapsed_precise}] {bar:40.green/yellow} {pos:>4}/{len:4}").unwrap();
    let sty_aux = ProgressStyle::with_template(
        "[{elapsed_precise}] {bar:20.cyan/blue} {pos:>8}/{len:8}B {msg}",
    )
    .unwrap()
    .progress_chars("##-");

    // - 1 to remove the metadata key
    let main_bar = ProgressBar::new(header_entries.len() as u64 - 1);
    main_bar.set_style(sty_main);
    let main_bar_clone = main_bar.clone();
    let mp = MultiProgress::new();
    mp.add(main_bar);

    // Process the header entries in parallel
    let processed_entries: Vec<(String, Value)> = layers
        .par_iter()
        .filter_map(|layer| {
            // Create the progress bar based on the length
            let pb = mp.add(ProgressBar::new(layer.size));
            pb.set_style(sty_aux.clone());
            pb.enable_steady_tick(Duration::from_millis(200));
            pb.set_message(format!("{}", layer.name));

            // Download the tensor
            // println!("{}: Downloading {}...", model_id, tensor_name);
            let tensor = download::download_tensor(
                url,
                layer.offset_start,
                layer.offset_end,
                Some(pb.clone())
            ).unwrap(); // Handle unwrap better
            // Hash the tensor
            let hash = hash::sha256_hash(&tensor);
            // Put that in the results
            let tensor_result = json!({
                "data_offsets": vec![layer.offset_start, layer.offset_end],
                "hash": hash
            });
            pb.finish_and_clear();
            main_bar_clone.inc(1);
            Some((layer.name.to_string(), tensor_result))
        
        })
        .collect();
    main_bar_clone.finish();
    mp.clear().unwrap();


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
