import requests
import json
import sys

# Replace with your actual API URL after running create_api_gateway.py
# Or pass as argument
API_URL = "" 

if len(sys.argv) > 1:
    API_URL = sys.argv[1]

if not API_URL:
    print("Please provide the API URL as an argument.")
    print("Usage: python test_api.py <API_URL>")
    sys.exit(1)

def test_prediction(text):
    payload = {"text": text}
    headers = {"Content-Type": "application/json"}
    
    print(f"Sending request to {API_URL} with text: '{text}'")
    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        print(f"Status Code: {response.status_code}")
        print("Response Body:")
        try:
            print(json.dumps(response.json(), indent=2))
        except:
            print(response.text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_prediction("I absolutely love this movie! It was fantastic.")
    print("-" * 20)
    test_prediction("This was a terrible waste of time. I hated it.")
