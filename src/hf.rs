use reqwest::{blocking::Client};
use serde_json;
use std::collections::HashMap;
use anyhow::{Error, Ok};

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, PartialEq)]
pub struct ModelInfo {
    id: String,
    author: Option<String>,
    sha: Option<String>,
    created_at: Option<String>, // Change this to appropriate date/time type if needed
    last_modified: Option<String>, // Change this to appropriate date/time type if needed
    private: bool,
    disabled: Option<bool>,
    gated: Option<String>, // Change this to appropriate type
    downloads: i32,
    likes: i32,
    library_name: Option<String>,
    tags: Vec<String>,
    pipeline_tag: Option<String>,
    mask_token: Option<String>,
    widget_data: Option<String>, // Change this to appropriate type
    model_index: Option<HashMap<String, String>>, // Change this to appropriate type
    config: Option<HashMap<String, String>>, // Change this to appropriate type
    // Add more fields as needed
}

fn fill_model_info_from_json(json_string: &str) -> Result<ModelInfo, Error> {
    let model_info: ModelInfo = serde_json::from_str(json_string)?;
    Ok(model_info)
}

pub fn get_model_info(model_id: &str) -> Result<ModelInfo, Error> {
    // TODO: setup headers?
    let client = Client::new();

    // TODO: handle non-main revisions in future
    let url = format!("https://huggingface.co/api/models/{}/revision/main", model_id);

    let response = client.get(url).send()?;

    let body = response.text().unwrap();

    // TODO: Handle malformed JSON better
    let result = fill_model_info_from_json(&body).unwrap();

    Ok(result)
}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
pub struct FileInfo {
    path: String,
    size: i64,
    blob_id: String,
    // TODO: Add all fields if necessary
    // lfs: BlobLfsInfo,
    // last_commit: Option<LastCommitInfo>,
    // security: Option<BlobSecurityInfo>,
}

fn fill_file_info_from_json(json_string: &str) -> Result<FileInfo, Error> {
    let file_info: FileInfo = serde_json::from_str(json_string)?;
    Ok(file_info)
}

pub fn get_model_files(model_id: &str) -> Result<FileInfo, Error> {
    // TODO: setup headers?
    let client = Client::new();

    // TODO: handle non-main revisions in future
    let url = format!("https://huggingface.co/api/models/{}/paths-info/main", model_id);

    let response = client.post(url).send()?;

    let body = response.text().unwrap();

    // TODO: Handle malformed JSON better
    // TODO: Handle multiple files, this isn't correct with the actual payload
    // See: https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/hf_api.py#L3017
    let result = fill_file_info_from_json(&body).unwrap();

    Ok(result)
}

mod tests {
    use super::*;

    #[test]
    fn test_fill_model_info_from_json() {
        // Mock JSON string for testing
        let json_string = r#"{
            "id": "model_id",
            "author": "author_name",
            "sha": "sha_value",
            "created_at": "2022-03-02T23:29:04.000Z",
            "last_modified": "2022-03-02T23:29:04.000Z",
            "private": false,
            "disabled": false,
            "gated": "auto",
            "downloads": 100,
            "likes": 50,
            "library_name": "library_name",
            "tags": ["tag1", "tag2"],
            "pipeline_tag": "pipeline_tag",
            "mask_token": "mask_token",
            "widget_data": "widget_data",
            "model_index": {"key": "value"},
            "config": {"key": "value"}
        }"#;

        // Expected ModelInfo struct
        let expected_model_info = ModelInfo {
            id: "model_id".to_string(),
            author: Some("author_name".to_string()),
            sha: Some("sha_value".to_string()),
            created_at: Some("2022-03-02T23:29:04.000Z".to_string()),
            last_modified: Some("2022-03-02T23:29:04.000Z".to_string()),
            private: false,
            disabled: Some(false),
            gated: Some("auto".to_string()),
            downloads: 100,
            likes: 50,
            library_name: Some("library_name".to_string()),
            tags: vec!["tag1".to_string(), "tag2".to_string()],
            pipeline_tag: Some("pipeline_tag".to_string()),
            mask_token: Some("mask_token".to_string()),
            widget_data: Some("widget_data".to_string()),
            model_index: Some(vec![("key".to_string(), "value".to_string())].into_iter().collect()),
            config: Some(vec![("key".to_string(), "value".to_string())].into_iter().collect()),
        };

        // Parse JSON and fill the struct
        let actual_model_info = fill_model_info_from_json(json_string).unwrap();

        // Compare expected with actual
        assert_eq!(expected_model_info, actual_model_info);
    }

    #[test]
    fn test_fill_file_info_from_json() {
        // Mock JSON string for testing
        let json_string = r#"{
            "path": "file/path/example.txt",
            "size": 1024,
            "blob_id": "abcdef123456"
        }"#;

        // Expected FileInfo struct
        let expected_file_info = FileInfo {
            path: "file/path/example.txt".to_string(),
            size: 1024,
            blob_id: "abcdef123456".to_string(),
        };

        // Parse JSON and fill the struct
        let actual_file_info: FileInfo = serde_json::from_str(json_string).unwrap();

        // Compare expected with actual
        assert_eq!(expected_file_info.path, actual_file_info.path);
        assert_eq!(expected_file_info.size, actual_file_info.size);
        assert_eq!(expected_file_info.blob_id, actual_file_info.blob_id);
    }
}
