# Dockerized Text Generation API
This project provides a Dockerized Text Generation API. It offers an HTTP API endpoint for generating text based on input prompts. The API allows you to customize text generation by specifying parameters like max_new_tokens, generate_type, no_repeat_ngram_size, and num_beams.

## Prerequisites
Before getting started, make sure you have the following installed on your system:
- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Usage
Follow the steps below to set up and use the text generation API:

### Using Docker Compose
1. Navigate to the project directory
2. Build and start the Docker containers:
  ```bash
    docker-compose up
  ```
3. Once the API is up and running, you can send a POST request to [http://localhost:8000/generate_text](http://localhost:8000/generate_text) to generate text.

### Running Locally

If you encounter CUDA-related errors when using Docker Compose, you can run the application locally:
1. Make sure you have the required Python packages installed. You can install them using `pip`:

    ```bash
    pip install -r requirements.txt
    ```
   or
    ```bash
    conda env create -f environment.yml 
   ```

2. Start the application by running:

    ```bash
    python main.py
    ```

3. The API will be accessible at [http://localhost:8000/generate_text](http://localhost:8000/generate_text).



## API Request
- **URL**: `http://localhost:8000/generate_text`
- **Method**: POST
- **Request Body**:
    ```json
    {
        "text": "what is your",
        "max_new_tokens": 5,
        "generate_type": "beam_search",
        "no_repeat_ngram_size": 1,
        "num_beams": 3
    }
    ```

## Example
You can use tools like `curl` or Postman to send requests. Here's a `curl` example:

```bash
curl -X POST "http://localhost:8000/generate_text" -H "Content-Type: application/json" -d '{
  "text": "what is your",
  "max_new_tokens": 5,
  "generate_type": "beam_search",
  "no_repeat_ngram_size": 1,
  "num_beams": 3
}'
```

## Configuration
You can customize the text generation by modifying the JSON payload mentioned in step 3. Here's what each parameter does:

- text: The starting text prompt for text generation.
- max_new_tokens: The maximum number of tokens in the generated text.
`generate
- generate_type: The generation method to use (e.g., "greedy_search, beam_search").
- no_repeat_ngram_size: The size of n-grams to avoid repeating in the generated text.
- num_beams: The number of beams to use in beam search.
