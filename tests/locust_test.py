from locust import HttpUser, task, between
from urllib.parse import quote

class TTSUser(HttpUser):
    wait_time = between(1, 3)  # Wait time between requests (in seconds)

    @task
    def tts_request(self):
        text = "So If you encounter permission errors while installing packages, you can try running PowerShell. "
        voice = "female_01.wav"
        language = "en"
        output_file = "output.wav"

        # Encode the text for URL
        encoded_text = quote(text)

        # Create the streaming URL
        streaming_url = f"/api/tts-generate-streaming?text={encoded_text}&voice={voice}&language={language}&output_file={output_file}"

        # Send a GET request to the streaming URL
        with self.client.get(streaming_url, stream=True, catch_response=True) as response:
            if response.status_code == 200:
                # Iterate over the response content chunks
                for chunk in response.iter_content(chunk_size=512):
                    pass
                response.success()
            else:
                response.failure(f"Request failed with status code: {response.status_code}")

# locust -f locust_test.py --host=https://7ba301f55a0c1.notebooksc.jarvislabs.net