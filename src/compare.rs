use std::fs::{self, File};
use std::io::{self, BufReader, Read, Seek, SeekFrom, Write};

use safetensors::SafeTensors;

pub fn compare_tensors_between_files(file_path_a: &str, file_path_b: &str) {
    let mut bytes_a = vec![];
    let mut f_a = fs::File::open(file_path_a).unwrap();
    f_a.read_to_end(&mut bytes_a).unwrap();
    let result_a = SafeTensors::deserialize(&bytes_a).unwrap();
    let mut bytes_b = vec![];
    let mut f_b = fs::File::open(file_path_a).unwrap();
    f_b.read_to_end(&mut bytes_b).unwrap();
    let result_b = SafeTensors::deserialize(&bytes_b).unwrap();
    for (name, tensor) in result_a.tensors() {
        println!("{} {:?}", name, tensor.data())
    }
    for (name, tensor) in result_b.tensors() {
        println!("{} {:?}", name, tensor.data())
    }
}