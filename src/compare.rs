use std::collections::HashMap;
use std::fs::{self};
use std::io::Read;

use safetensors::{SafeTensors, View};

use crate::{hasher, hf};

pub fn compare_hashes_via_registry(model_id_a: &str, model_id_b: &str) {
    let model_a_layers_to_hashes = get_model_hashes_by_model_id(model_id_a);
    let model_b_layers_to_hashes = get_model_hashes_by_model_id(model_id_b);

    // Extract the hashes
    let model_a_hashes: Vec<String> = model_a_layers_to_hashes
        .iter()
        .map(|(_, v)| v.to_string())
        .collect();
    let model_b_hashes: Vec<String> = model_b_layers_to_hashes
        .iter()
        .map(|(_, v)| v.to_string())
        .collect();

    let mut same_hash_counter = 0;
    for hash_a in model_a_hashes.iter() {
        if model_b_hashes.contains(&hash_a) {
            same_hash_counter += 1;
        }
    }
    println!(
        "{}: {} layers compared to {}: {} layers",
        model_id_a,
        model_a_hashes.len(),
        model_id_b,
        model_b_hashes.len()
    );
    println!("{} layers found in common.", same_hash_counter);
}

fn get_model_hashes_by_model_id(model_id: &str) -> HashMap<String, String> {
    // Get all safetensor file names

    // Query the HF API to see the file names
    let model_info_result = hf::get_model_info(model_id);
    if model_info_result.is_err() {
        // TODO: Handle better, print the error message too
        panic!("Unable to retrieve model info when attempting download!")
    }

    let model_info = model_info_result.unwrap();

    let model_filenames: Vec<&String> = model_info
        .siblings
        .as_slice()
        .iter()
        .map(|s| &s.rfilename)
        .collect();

    let safetensors_filenames: Vec<&String> = model_filenames
        .into_iter()
        .filter(|mf| mf.ends_with(".safetensors"))
        .collect();

    // Create a Map to hold all tensor names to hashes
    let mut all_layers_to_hashes: HashMap<String, String> = HashMap::new();

    // Get the hashes of each file
    for file_name in safetensors_filenames {
        let (_, layers_to_hashes_map) = hasher::get_model_file_hashes(model_id, file_name);

        for result in layers_to_hashes_map.iter() {
            all_layers_to_hashes.insert(result.0.to_string(), result.1.to_string());
        }
    }

    all_layers_to_hashes
}

pub fn compare_tensors_between_files(file_path_a: &str, file_path_b: &str) {
    let mut bytes_a = vec![];
    let mut f_a = fs::File::open(file_path_a).unwrap();
    f_a.read_to_end(&mut bytes_a).unwrap();
    let result_a = SafeTensors::deserialize(&bytes_a).unwrap();
    let mut bytes_b = vec![];
    let mut f_b = fs::File::open(file_path_a).unwrap();
    f_b.read_to_end(&mut bytes_b).unwrap();
    let result_b = SafeTensors::deserialize(&bytes_b).unwrap();

    // let name_to_tensor_map_a:HashMap<String, TensorView> = result_a.tensors().into_iter().collect();

    let mut all_layers_same = true;
    for (name, tensor_a) in result_a.tensors() {
        println!("Comparing tensors with the name: {}", name);
        // println!("{:?}", tensor.data());
        let tensor_b = result_b.tensor(&name).unwrap();
        let (count_same, all_same) = compare_slices(tensor_a.data(), tensor_b.data());
        println!(
            "Count Same: {}/{}, All Same: {}",
            count_same,
            tensor_a.data_len(),
            all_same
        );
        if !all_same {
            all_layers_same = false;
        }
    }

    println!(
        "The tensors of {} and {} and completely the same at a byte level: {}",
        file_path_a, file_path_b, all_layers_same
    )
    // for (name, tensor_b) in result_b.tensors() {
    //     println!("{}", name);
    //     // println!("{:?}", tensor.data());
    //     break;

    // }
}

fn compare_slices(slice1: &[u8], slice2: &[u8]) -> (usize, bool) {
    if slice1.len() != slice2.len() {
        return (0, false);
    }

    let mut count_same = 0;
    let mut all_same = true;

    for (elem1, elem2) in slice1.iter().zip(slice2.iter()) {
        if elem1 == elem2 {
            count_same += 1;
        } else {
            all_same = false;
        }
    }

    (count_same, all_same)
}
