# Configuration

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**chunk_processing** | [**models::ChunkProcessing**](ChunkProcessing.md) |  | 
**error_handling** | [**models::ErrorHandlingStrategy**](ErrorHandlingStrategy.md) |  | 
**expires_in** | Option<**i32**> | The number of seconds until task is deleted. Expried tasks can **not** be updated, polled or accessed via web interface. | [optional]
**high_resolution** | **bool** | Whether to use high-resolution images for cropping and post-processing. | 
**input_file_url** | Option<**String**> | The presigned URL of the input file. | [optional]
**json_schema** | Option<[**serde_json::Value**](.md)> |  | [optional]
**llm_processing** | [**models::LlmProcessing**](LlmProcessing.md) |  | 
**model** | Option<[**models::Model**](Model.md)> |  | [optional]
**ocr_strategy** | [**models::OcrStrategy**](OcrStrategy.md) |  | 
**segment_processing** | [**models::SegmentProcessing**](SegmentProcessing.md) |  | 
**segmentation_strategy** | [**models::SegmentationStrategy**](SegmentationStrategy.md) |  | 
**target_chunk_length** | Option<**i32**> | The target number of words in each chunk. If 0, each chunk will contain a single segment. | [optional]

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


