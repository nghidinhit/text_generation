from decouple import config
from fastapi import APIRouter, HTTPException

import constants
from models import Prompt
from error import CustomError
from logger import LoggerSingleton
from services import Generator

logger = LoggerSingleton("api").logger

api = APIRouter()
generator = Generator(config("MODEL_NAME"))


@api.post("/generate_text")
async def generate_text(prompt: Prompt):
    try:
        if prompt.generate_type == constants.GENERATE_TYPE_GREEDY or prompt.num_beams == 1:
            generated_text = generator.generate_greedy_search(prompt.text, prompt.max_new_tokens, prompt.no_repeat_ngram_size)
        elif prompt.generate_type == constants.GENERATE_TYPE_BEAM_SEARCH:
            generated_text = generator.generate_beam_search(prompt.text, prompt.max_new_tokens, prompt.no_repeat_ngram_size, prompt.num_beams)
        else:
            raise CustomError(constants.ERROR_MSG_INVALID_GENERATE_TYPE, status_code=constants.ERROR_NO_400)

        return {"text": generated_text}
    except CustomError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=constants.ERROR_NO_500, detail=str(e))
