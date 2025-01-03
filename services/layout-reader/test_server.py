import requests
import json
import time

def test_layout_reader(url="http://localhost:8000"):
    # Sample input data
    payload = {
        "text": "Hello World Example Text",
        "bboxes": [
            [120, 0, 200, 50],  # World
            [0, 0, 100, 50],    # Hello
            [220, 0, 300, 50],  # Example
            [320, 0, 400, 50]   # Text
        ]
    }

    # Send POST request to /predict endpoint
    try:
        start_time = time.time()
        response = requests.post(
            f"{url}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        request_time = time.time() - start_time
        print(f"Request took {request_time:.3f} seconds")
        
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            print("Error Response:", response.text)
            raise e
        
        # Print results
        result = response.json()
        print("Reading order:", result["reading_order"])
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None

    except json.JSONDecodeError:
        print("Error decoding response")
        return None

if __name__ == "__main__":
    # Test the health endpoint first
    try:
        health = requests.get("http://localhost:8000")
        print("Health check:", health.json())
    except:
        print("Server might not be running!")
        exit(1)
    
    # Test the prediction endpoint
    result = test_layout_reader()