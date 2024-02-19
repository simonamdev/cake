use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufReader, Read, Seek, SeekFrom, Write};

use safetensors::tensor::TensorView;
use safetensors::{SafeTensors, View};

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
        let (count_same, all_same) = compare_slices(
            tensor_a.data(), 
            tensor_b.data()
        );
        println!("Count Same: {}/{}, All Same: {}", count_same, tensor_a.data_len(), all_same);
        if !all_same {
            all_layers_same = false;
        }
    }

    println!("The tensors of {} and {} and completely the same at a byte level: {}", file_path_a, file_path_b, all_layers_same)
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
