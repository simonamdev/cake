use sha2::{Sha256, Digest};

pub fn sha256_hash(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let result = hasher.finalize();

    let hash_hex = format!("{:x}", result);

    hash_hex
}