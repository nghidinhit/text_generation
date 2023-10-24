from pydantic import BaseModel


class Prompt(BaseModel):
    text: str
    max_new_tokens: int = 32
    no_repeat_ngram_size: int = 0
    generate_type: str
    num_beams: int = 1
