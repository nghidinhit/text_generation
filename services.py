import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from logger import LoggerSingleton

logger = LoggerSingleton("services").logger


class Generator:
    def __init__(self, model_name):
        """
        Initialize an instance of Generator.
        Args:
            model_name (str): The name of the model you want to use.
        """
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f'Device: {self.device}')
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.device == "cpu":
                self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=self.device, torch_dtype=torch.float32, low_cpu_mem_usage=True)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=self.device, torch_dtype="auto")
            self.vocab_size = self.tokenizer.vocab_size
        except Exception as e:
            logger.info(f"An error occurred during model initialization: {str(e)}")
            self.device = "cpu"

    def encode(self, text):
        """
        Convert a text to tokens and convert tokens to IDs.
        Args:
            text (str): The input text.
        Returns:
            torch.Tensor: Tensors containing IDs of the tokens.
        """
        return self.tokenizer.encode(text, return_tensors="pt").to(self.device)

    def decode(self, ids):
        """
        Convert IDs to text.
        Args:
            ids (torch.Tensor): Tensors containing IDs of the tokens.
        Returns:
            str: Text generated from IDs.
        """
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @staticmethod
    def get_top1(prob):
        """
        Get the ID of the token with the highest probability.
        Args:
            prob (torch.Tensor): Probabilities of tokens.
        Returns:
            torch.Tensor: Score of the token with the highest probability.
            torch.Tensor: ID of the token with the highest probability.
        """

        score, token_id = torch.max(prob, dim=-1)
        return score, token_id

    @staticmethod
    def get_topk(prob, k=1):
        """
        Get the top k tokens with the highest probabilities.
        Args:
            prob (torch.Tensor): Probabilities of tokens.
            k (int): Number of tokens to retrieve.
        Returns:
            torch.Tensor: Scores of the top k tokens.
            torch.Tensor: IDs of the top k tokens.
        """
        scores, token_ids = torch.topk(prob, k=k, dim=-1)
        return scores, token_ids

    def get_next_token_prob(self, input_ids: torch.Tensor):
        """
        Get the probability of the next token for a given input.
        Args:
            input_ids (torch.Tensor): Input tokens.

        Returns:
            torch.Tensor: Probability of all tokens for the next position.
        """
        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits
        next_token_prob = logits[:, -1, :]
        return next_token_prob

    @staticmethod
    def calc_banned_ngram_tokens(prev_input_ids, num_hypos, no_repeat_ngram_size, cur_len):
        """
        Calculate banned n-gram tokens to prevent repetitions.

        Args:
            prev_input_ids (torch.Tensor): Previous input tokens.
            num_hypos (int): Number of hypotheses.
            no_repeat_ngram_size (int): Size of n-gram to check.
            cur_len (int): Current length of input.

        Returns:
            List[List[int]]: List of banned tokens for each hypothesis.
        """
        if cur_len + 1 < no_repeat_ngram_size:
            return [[] for _ in range(num_hypos)]

        generated_ngrams = [{} for _ in range(num_hypos)]

        for hypo_idx in range(num_hypos):
            gen_tokens = prev_input_ids[hypo_idx]
            generated_ngram = generated_ngrams[hypo_idx]

            ngrams = [gen_tokens[i:i + no_repeat_ngram_size] for i in range(cur_len - no_repeat_ngram_size + 1)]

            for ngram in ngrams:
                prev_ngram_tuple = tuple(ngram[:-1])
                generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

        def get_generated_ngrams(hypo_idx):
            start_idx = cur_len + 1 - no_repeat_ngram_size
            ngram_idx = tuple(prev_input_ids[hypo_idx][start_idx:cur_len])
            return generated_ngrams[hypo_idx].get(ngram_idx, [])

        banned_tokens = [get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]

        return banned_tokens

    def post_process(self, input_ids, batch_size, num_beams, cur_len, next_token_logits, no_repeat_ngram_size):
        """
        Apply post-processing to modify token probabilities.

        Args:
            input_ids (torch.Tensor): Input tokens.
            batch_size (int): Batch size.
            num_beams (int): Number of beams.
            cur_len (int): Current length of input.
            next_token_logits (torch.Tensor): Logits for the next token.
            no_repeat_ngram_size (int): Size of n-gram to check for repetition.

        Returns:
            torch.Tensor: Modified token probabilities.
        """
        if no_repeat_ngram_size > 0:
            num_batch_hypotheses = batch_size * num_beams
            banned_batch_tokens = self.calc_banned_ngram_tokens(
                input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
            )
            for i, banned_tokens in enumerate(banned_batch_tokens):
                logger.info(banned_tokens)
                next_token_logits[i, banned_tokens] = -float("inf")

        return next_token_logits

    def generate_greedy_search(self, prompt, max_new_tokens=32, no_repeat_ngram_size=0):
        """
        Generate text using greedy search strategy.

        Args:
            prompt (str): The input text prompt.
            max_new_tokens (int): Maximum number of tokens to generate.
            no_repeat_ngram_size (int): Size of n-gram to check for repetition.

        Returns:
            str: Generated text.
        """
        try:
            input_ids = self.encode(prompt)
            batch_size = input_ids.shape[0]
            cur_len = input_ids.shape[-1]
            max_length = cur_len + max_new_tokens

            while cur_len < max_length:
                next_token_prob = self.get_next_token_prob(input_ids)
                next_token_prob = self.post_process(input_ids, batch_size, 1, cur_len, next_token_prob, no_repeat_ngram_size)
                score, token_id = self.get_top1(next_token_prob)

                if token_id.item() == self.tokenizer.eos_token_id:
                    break

                input_ids = torch.cat((input_ids, token_id.unsqueeze(-1)), dim=-1)
                cur_len += 1

            output_text = self.decode(input_ids[0])
            return output_text
        except Exception as e:
            logger.info(f"An error occurred during greedy search: {str(e)}")

    def generate_beam_search(self, prompt, max_new_tokens=32, no_repeat_ngram_size=0, num_beams=1):
        """
            Generate text using beam search based on the provided prompt.

            Args:
                prompt (str): The initial text prompt to start text generation.
                max_new_tokens (int): The maximum number of tokens to generate.
                no_repeat_ngram_size (int): If greater than 0, it prevents ngrams of this size from repeating in the output.
                num_beams (int): The number of beams for the beam search algorithm.

            Returns:
                List[str]: A list of generated texts using beam search.

        """
        try:
            input_ids = self.encode(prompt)

            batch_size = input_ids.shape[0]
            cur_len = input_ids.shape[-1]
            max_length = cur_len + max_new_tokens

            # Tạo các tensor để lưu trữ thông tin cho mỗi beam
            beam_input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, cur_len)
            beam_input_ids = beam_input_ids.contiguous().view(batch_size * num_beams, cur_len)
            beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
            beam_scores = beam_scores.view(-1)

            while cur_len < max_length:
                next_token_prob = self.get_next_token_prob(beam_input_ids)
                next_token_prob = F.log_softmax(next_token_prob, dim=-1)  # (batch_size * num_beams, vocab_size)
                next_token_prob = self.post_process(beam_input_ids, batch_size, num_beams, cur_len, next_token_prob,
                                                    no_repeat_ngram_size)

                next_scores = next_token_prob + beam_scores[:, None].expand_as(next_token_prob)  # (bsz * num_beam, vocab_size)
                next_scores = next_scores.view(batch_size, -1)      # (bsz, num_beams * vocab_size)

                # Lựa chọn top 2 * num_beams tokens
                next_scores, next_tokens = torch.topk(next_scores, 2*num_beams, dim=1, largest=True, sorted=True)  # (bsz, 2*num_beams)
                # for token in next_tokens:
                #     print(f'top_k token {token}: {self.decode(token)}')

                # Tạo các tensor mới để lưu trữ thông tin cho mỗi beam sau khi mở rộng
                new_beam_input_ids = torch.zeros((batch_size, num_beams, cur_len + 1), dtype=torch.long,
                                                 device=input_ids.device)
                for batch_idx in range(batch_size):
                   for beam_id in range(num_beams):
                        new_beam_input_ids[batch_idx, beam_id, :cur_len] = beam_input_ids.unsqueeze(0)[batch_idx, beam_id, :]
                # new_beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
                new_beam_scores = torch.empty((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
                new_beam_scores.fill_(float("-inf"))

                for batch_idx in range(batch_size):
                    for beam_idx in range(2*num_beams):
                        token_idx = next_tokens[batch_idx, beam_idx].item()

                        beam_id = token_idx // self.vocab_size
                        token_id = token_idx % self.vocab_size

                        if next_scores[batch_idx, beam_idx] > new_beam_scores[batch_idx, beam_id]:
                            new_beam_input_ids[batch_idx, beam_id, cur_len] = token_id
                            new_beam_scores[batch_idx, beam_id] = next_scores[batch_idx, beam_idx]

                beam_input_ids = new_beam_input_ids.view(batch_size * num_beams, -1)
                beam_scores = new_beam_scores.view(-1)

                cur_len += 1

            output_texts = [self.decode(ids) for ids in beam_input_ids]
            for i, text in enumerate(output_texts):
                logger.info(f"Beam {i + 1}: {text}")

            return output_texts
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.info(f"An error occurred during beam search: {str(e)}")
