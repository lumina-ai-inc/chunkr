# UpdateForm

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**chunk_processing** | Option<[**models::ChunkProcessing**](ChunkProcessing.md)> |  | [optional]
**error_handling** | Option<[**models::ErrorHandlingStrategy**](ErrorHandlingStrategy.md)> |  | [optional]
**expires_in** | Option<**i32**> | The number of seconds until task is deleted. Expired tasks can **not** be updated, polled or accessed via web interface. | [optional]
**high_resolution** | Option<**bool**> | Whether to use high-resolution images for cropping and post-processing. (Latency penalty: ~7 seconds per page) | [optional]
**llm_processing** | Option<[**models::LlmProcessing**](LlmProcessing.md)> |  | [optional]
**ocr_strategy** | Option<[**models::OcrStrategy**](OcrStrategy.md)> |  | [optional]
**segment_processing** | Option<[**models::SegmentProcessing**](SegmentProcessing.md)> |  | [optional]
**segmentation_strategy** | Option<[**models::SegmentationStrategy**](SegmentationStrategy.md)> |  | [optional]

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


