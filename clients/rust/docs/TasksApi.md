# \TasksApi

All URIs are relative to *https://api.chunkr.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_tasks_route**](TasksApi.md#get_tasks_route) | **GET** /api/v1/tasks | Get Tasks



## get_tasks_route

> Vec<models::TaskResponse> get_tasks_route(base64_urls, end, include_chunks, limit, page, start)
Get Tasks

Retrieves a list of tasks  Example usage: `GET /api/v1/tasks?page=1&limit=10&include_chunks=false`

### Parameters


Name | Type | Description  | Required | Notes
------------- | ------------- | ------------- | ------------- | -------------
**base64_urls** | Option<**bool**> | Whether to return base64 encoded URLs. If false, the URLs will be returned as presigned URLs. |  |
**end** | Option<**String**> | End date |  |
**include_chunks** | Option<**bool**> | Whether to include chunks in the output response |  |
**limit** | Option<**i64**> | Number of tasks per page |  |
**page** | Option<**i64**> | Page number |  |
**start** | Option<**String**> | Start date |  |

### Return type

[**Vec<models::TaskResponse>**](TaskResponse.md)

### Authorization

[api_key](../README.md#api_key)

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json, text/plain

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

