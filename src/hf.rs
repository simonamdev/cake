use anyhow::{Error, Ok};
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use serde_json::{self};
use std::env;
use std::result::Result::Ok as stdOk;

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, PartialEq)]
pub struct ModelInfo {
    id: String,
    author: Option<String>,
    sha: Option<String>,
    private: bool,
    disabled: Option<bool>,
    downloads: i32,
    likes: i32,
    library_name: Option<String>,
    tags: Vec<String>,
    pipeline_tag: Option<String>,
    pub siblings: Vec<Sibling>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Sibling {
    pub rfilename: String,
}

fn fill_model_info_from_json(json_string: &str) -> Result<ModelInfo, serde_json::Error> {
    let mut model_info: ModelInfo = serde_json::from_str(json_string)?;

    // Ensure the siblings field exists in the JSON
    if let stdOk(value) = serde_json::from_str::<serde_json::Value>(json_string) {
        if let Some(siblings) = value.get("siblings") {
            if let Some(siblings) = siblings.as_array() {
                model_info.siblings = siblings
                    .iter()
                    .filter_map(|sibling| serde_json::from_value::<Sibling>(sibling.clone()).ok())
                    .collect();
            }
        }
    }

    stdOk(model_info)
}

pub fn get_model_info(model_id: &str) -> Result<ModelInfo, Error> {
    // TODO: setup headers?
    let client = Client::new();

    // TODO: handle non-main revisions in future
    let url = format!(
        "https://huggingface.co/api/models/{}/revision/main",
        model_id
    );

    // Set up headers
    let mut headers = HeaderMap::new();

    if let stdOk(bearer_token) = env::var("HF_API_KEY") {
        let bearer_value = format!("Bearer {}", bearer_token);
        headers.insert(AUTHORIZATION, HeaderValue::from_str(&bearer_value).unwrap());
    }

    let response = client.get(url).headers(headers).send()?;

    let body = response.text().unwrap();
    println!("{}", body);

    // TODO: Handle malformed JSON better
    let result = fill_model_info_from_json(&body).unwrap();

    Ok(result)
}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
pub struct FileInfo {
    pub path: String,
    size: i64,
    blob_id: String,
    // TODO: Add all fields if necessary
    // lfs: BlobLfsInfo,
    // last_commit: Option<LastCommitInfo>,
    // security: Option<BlobSecurityInfo>,
}

// TODO: Support FolderInfo
fn _fill_file_info_from_json(json_string: &str) -> Result<Vec<FileInfo>, Error> {
    let file_infos: Vec<FileInfo> = serde_json::from_str(json_string)?;
    Ok(file_infos)
}

pub fn _get_model_files(model_id: &str) -> Result<Vec<FileInfo>, Error> {
    // TODO: setup headers?
    let client = Client::new();

    // TODO: handle non-main revisions in future
    let url = format!(
        "https://huggingface.co/api/models/{}/paths-info/main",
        model_id
    );
    // TOOD: pass in the paths required here: seems we need to get that from the tree url
    // Looks like passing a glob of *.safetensors does not work. We can look at how huggingface-cli doe sit
    // See: https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/hf_api.py#L2793C10-L2794C103

    // Test URL: https://huggingface.co/KoboldAI/fairseq-dense-1.3B
    let payload = serde_json::json!({
        "paths": ["*.safetensors"]
    });

    let response = client.post(url).json(&payload).send()?;

    let body = response.text().unwrap();

    // TODO: Handle malformed JSON better
    // TODO: Handle multiple files, this isn't correct with the actual payload
    // See: https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/hf_api.py#L3017
    println!("{:?}", body);

    let result = _fill_file_info_from_json(&body).unwrap();

    Ok(result)
}

mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_fill_model_info_from_json() {
        // Mock JSON string for testing
        let json_string = r#"{
            "id": "model_id",
            "author": "author_name",
            "sha": "sha_value",
            "private": false,
            "disabled": false,
            "downloads": 100,
            "likes": 50,
            "library_name": "library_name",
            "tags": ["tag1", "tag2"],
            "pipeline_tag": "pipeline_tag",
            "siblings": [{"rfilename": "foo.safetensors"}]
        }"#;

        // Expected ModelInfo struct
        let expected_model_info = ModelInfo {
            id: "model_id".to_string(),
            author: Some("author_name".to_string()),
            sha: Some("sha_value".to_string()),
            private: false,
            disabled: Some(false),
            downloads: 100,
            likes: 50,
            library_name: Some("library_name".to_string()),
            tags: vec!["tag1".to_string(), "tag2".to_string()],
            pipeline_tag: Some("pipeline_tag".to_string()),
            siblings: vec![Sibling {
                rfilename: "foo.safetensors".to_string(),
            }],
        };

        // Parse JSON and fill the struct
        let actual_model_info = fill_model_info_from_json(json_string).unwrap();

        // Compare expected with actual
        assert_eq!(expected_model_info, actual_model_info);
    }

    #[test]
    fn test_fill_model_info_with_real_example_from_json() {
        // Mock JSON string for testing
        let json_string = r#"{
            "_id": "656b7fec4ab7bc884d6c9143",
            "id": "TheBloke/DiscoLM-120b-GPTQ",
            "modelId": "TheBloke/DiscoLM-120b-GPTQ",
            "author": "TheBloke",
            "sha": "d7a8e6b389680aab92fa9ec3d33067a7d7a35cd0",
            "lastModified": "2023-12-03T10:31:01.000Z",
            "private": false,
            "disabled": false,
            "pipeline_tag": "text-generation",
            "tags": [
                "transformers",
                "safetensors"
            ],
            "downloads": 5,
            "library_name": "transformers",
            "likes": 3,
            "model-index": null,
            "config": {
                "architectures": [
                    "LlamaForCausalLM"
                ],
                "model_type": "llama",
                "quantization_config": {
                    "bits": 4
                },
                "tokenizer_config": {
                    "bos_token": "<s>",
                    "eos_token": "<|im_end|>",
                    "pad_token": "</s>",
                    "unk_token": "<unk>",
                    "use_default_system_prompt": true
                }
            },
            "cardData": {
                "base_model": "DiscoResearch/DiscoLM-120b",
                "datasets": [
                    "Open-Orca/SlimOrca-Dedup",
                    "teknium/openhermes",
                    "meta-math/MetaMathQA",
                    "migtissera/Synthia-v1.3",
                    "THUDM/AgentInstruct",
                    "LeoLM/German_Songs",
                    "LeoLM/German_Poems",
                    "LeoLM/OpenSchnabeltier",
                    "bjoernp/ultrachat_de"
                ],
                "inference": false,
                "language": [
                    "en"
                ],
                "library_name": "transformers",
                "license": "llama2",
                "model_creator": "Disco Research",
                "model_name": "DiscoLM 120B",
                "model_type": "llama",
                "pipeline_tag": "text-generation",
                "prompt_template": "<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                "quantized_by": "TheBloke",
                "tags": [
                    "goliath",
                    "deutsch",
                    "llama2",
                    "discoresearch"
                ]
            },
            "transformersInfo": {
                "auto_model": "AutoModelForCausalLM",
                "pipeline_tag": "text-generation",
                "processor": "AutoTokenizer"
            },
            "spaces": [],
            "siblings": [
                {
                    "rfilename": ".gitattributes"
                },
                {
                    "rfilename": "LICENSE"
                },
                {
                    "rfilename": "Notice"
                },
                {
                    "rfilename": "README.md"
                },
                {
                    "rfilename": "added_tokens.json"
                },
                {
                    "rfilename": "config.json"
                },
                {
                    "rfilename": "generation_config.json"
                },
                {
                    "rfilename": "model-00001-of-00006.safetensors"
                },
                {
                    "rfilename": "model-00002-of-00006.safetensors"
                },
                {
                    "rfilename": "model-00003-of-00006.safetensors"
                },
                {
                    "rfilename": "model-00004-of-00006.safetensors"
                },
                {
                    "rfilename": "model-00005-of-00006.safetensors"
                },
                {
                    "rfilename": "model-00006-of-00006.safetensors"
                },
                {
                    "rfilename": "model.safetensors.index.json"
                },
                {
                    "rfilename": "quantize_config.json"
                },
                {
                    "rfilename": "special_tokens_map.json"
                },
                {
                    "rfilename": "tokenizer.model"
                },
                {
                    "rfilename": "tokenizer_config.json"
                }
            ],
            "safetensors": {
                "parameters": {
                    "I32": 14664900864,
                    "F16": 550072320
                },
                "total": 15214973184
            }
        }"#;

        // Expected ModelInfo struct
        let expected_model_info = ModelInfo {
            id: "TheBloke/DiscoLM-120b-GPTQ".to_string(),
            author: Some("TheBloke".to_string()),
            sha: Some("d7a8e6b389680aab92fa9ec3d33067a7d7a35cd0".to_string()),
            private: false,
            disabled: Some(false),
            downloads: 5,
            likes: 3,
            library_name: Some("transformers".to_string()),
            tags: vec!["transformers".to_string(), "safetensors".to_string()],
            pipeline_tag: Some("text-generation".to_string()),
            siblings: vec![
                Sibling {
                    rfilename: ".gitattributes".to_string(),
                },
                Sibling {
                    rfilename: "LICENSE".to_string(),
                },
                Sibling {
                    rfilename: "Notice".to_string(),
                },
                Sibling {
                    rfilename: "README.md".to_string(),
                },
                Sibling {
                    rfilename: "added_tokens.json".to_string(),
                },
                Sibling {
                    rfilename: "config.json".to_string(),
                },
                Sibling {
                    rfilename: "generation_config.json".to_string(),
                },
                Sibling {
                    rfilename: "model-00001-of-00006.safetensors".to_string(),
                },
                Sibling {
                    rfilename: "model-00002-of-00006.safetensors".to_string(),
                },
                Sibling {
                    rfilename: "model-00003-of-00006.safetensors".to_string(),
                },
                Sibling {
                    rfilename: "model-00004-of-00006.safetensors".to_string(),
                },
                Sibling {
                    rfilename: "model-00005-of-00006.safetensors".to_string(),
                },
                Sibling {
                    rfilename: "model-00006-of-00006.safetensors".to_string(),
                },
                Sibling {
                    rfilename: "model.safetensors.index.json".to_string(),
                },
                Sibling {
                    rfilename: "quantize_config.json".to_string(),
                },
                Sibling {
                    rfilename: "special_tokens_map.json".to_string(),
                },
                Sibling {
                    rfilename: "tokenizer.model".to_string(),
                },
                Sibling {
                    rfilename: "tokenizer_config.json".to_string(),
                },
            ],
        };

        // Parse JSON and fill the struct
        let actual_model_info = fill_model_info_from_json(json_string).unwrap();

        // Compare expected with actual
        assert_eq!(expected_model_info, actual_model_info);
    }

    #[test]
    fn test_fill_file_info_from_json() {
        // Mock JSON string for testing
        let json_string = r#"[
            {
                "path": "file/path/example1.txt",
                "size": 1024,
                "blob_id": "abcdef123456"
            },
            {
                "path": "file/path/example2.txt",
                "size": 2048,
                "blob_id": "123456abcdef"
            }
        ]"#;

        // Expected FileInfo struct
        let expected_file_infos = [
            FileInfo {
                path: "file/path/example1.txt".to_string(),
                size: 1024,
                blob_id: "abcdef123456".to_string(),
            },
            FileInfo {
                path: "file/path/example2.txt".to_string(),
                size: 2048,
                blob_id: "123456abcdef".to_string(),
            },
        ];

        // Parse JSON and fill the struct
        let actual_file_infos: Vec<FileInfo> = serde_json::from_str(json_string).unwrap();

        // Compare expected with actual
        assert_eq!(expected_file_infos[0].path, actual_file_infos[0].path);
        assert_eq!(expected_file_infos[0].size, actual_file_infos[0].size);
        assert_eq!(expected_file_infos[0].blob_id, actual_file_infos[0].blob_id);

        assert_eq!(expected_file_infos[1].path, actual_file_infos[1].path);
        assert_eq!(expected_file_infos[1].size, actual_file_infos[1].size);
        assert_eq!(expected_file_infos[1].blob_id, actual_file_infos[1].blob_id);
    }
}
