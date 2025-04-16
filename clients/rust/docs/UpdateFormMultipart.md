# UpdateFormMultipart

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**chunk_processing** | Option<[**models::UpdateFormMultipartChunkProcessing**](UpdateFormMultipart_chunk_processing.md)> |  | [optional]
**expires_in** | Option<**i32**> | The number of seconds until task is deleted. Expried tasks can **not** be updated, polled or accessed via web interface. | [optional]
**high_resolution** | Option<**bool**> | Whether to use high-resolution images for cropping and post-processing. (Latency penalty: ~7 seconds per page) | [optional]
**ocr_strategy** | Option<[**models::OcrStrategy**](OcrStrategy.md)> |  | [optional]
**segment_processing** | Option<[**models::UpdateFormMultipartSegmentProcessing**](UpdateFormMultipart_segment_processing.md)> |  | [optional]
**segmentation_strategy** | Option<[**models::SegmentationStrategy**](SegmentationStrategy.md)> |  | [optional]

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


