# Pretty much same as querying openAI because vllm hosts openAI compatible severs

import argparse
import asyncio
import logging

import aiolimiter
import openai
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# Reference:
# https://github.com/zeno-ml/zeno-build/blob/3a5dcb2ed8bfdeec0dd2bacf2bea76673399824d/zeno_build/models/providers/openai_utils.py#L118

# Edit: Recent changes (11-15-2023) prompted by various changes to the OpenAI API.


ERROR_ERRORS_TO_MESSAGES = {
    openai.AuthenticationError: "Authentication Error: {e}. Usually due to incorrect OPENAI_API_KEY.",
    openai.BadRequestError: "Bad Request Error: {e}. Usually due to missing or invalid parameters.",
    openai.PermissionDeniedError: "Permission Denied Error: {e}. You don't have access to the requested resource.",
    openai.NotFoundError: "Not Found Error: {e}. The requested resource doesn't exist.",
    openai.APIConnectionError: "API Connection Error: {e}. There was a problem connecting to the OpenAI API.",
    openai.APITimeoutError: "API Timeout Error: {e}. The request timed out.",
    openai.InternalServerError: "Internal Server Error: {e}. An error occurred on the server side.",
    openai.RateLimitError: "Rate Limit Error: {e}. You've hit the OpenAI API rate limit.",
    openai.UnprocessableEntityError: "Unprocessable Entity Error: {e}. Unable to process the request despite the format being correct.",
}


def format_prompt(prompt_input):
    # prompt = prompt_input.split("\t")[-1]
    # return [{"role": "user", "content": prompt}]
    return [{'role': 'system', 'content': prompt_input[0]}, {'role': 'user', 'content': prompt_input[1]}]


def format_prompt_gemma(prompt_input):
    return[{'role': 'user', 'content': prompt_input[0] + '\n' + prompt_input[1]}]


async def _throttled_openai_chat_completion_acreate(
    client,
    model,
    messages,
    temperature,
    max_tokens,
    top_p,
    limiter: aiolimiter.AsyncLimiter,
    num_responses_per_prompt,
):
    async with limiter:
        for _ in range(3):
            try:
                return await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    n=num_responses_per_prompt,
                )
            except (
                openai.AuthenticationError,
                openai.BadRequestError,
                openai.PermissionDeniedError,
                openai.NotFoundError,
            ) as e:
                logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e))
            except openai.UnprocessableEntityError as e:
                logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e))
                return {
                    "choices": [
                        {"message": {"content": "Invalid Request: Prompt was filtered"}}
                    ]
                }
            except (
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.InternalServerError,
            ) as e:
                logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e))
                await asyncio.sleep(10)
            except openai.RateLimitError as e:
                logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e))
                await asyncio.sleep(60)
            except Exception as e:
                logging.warning(e)
            await asyncio.sleep(30)
        return {"choices": [{"message": {"content": ""}}]}


async def generate_from_openai_chat_completion(
    client,
    messages_list,
    model,
    temperature,
    max_tokens,
    top_p,
    requests_per_minute,
    num_responses_per_prompt,
) -> list[list[str]]:
    """Generate from OpenAI Chat Completion API.

    Args:
        prompts: Prompts.
        model: Model.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        requests_per_minute: Number of requests per minute to allow.
        num_responses_per_prompt: Number of responses to generate per prompt.

    Returns:
        List of generated responses.
    """
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            client=client,
            model=model,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
            num_responses_per_prompt=num_responses_per_prompt,
        )
        for message in messages_list
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    return responses


def vllm_inference(
    client,
    prompts,
    model,
    temperature,
    max_tokens,
    top_p,
    requests_per_minute,
    num_responses_per_prompt,
):
    if 'gemma' in model:
        messages_list = [format_prompt_gemma(prompt) for prompt in prompts]
    else:
        messages_list = [format_prompt(prompt) for prompt in prompts]
    predictions = asyncio.run(
        generate_from_openai_chat_completion(
            client=client,
            messages_list=messages_list,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            requests_per_minute=requests_per_minute,
            num_responses_per_prompt=num_responses_per_prompt,
        )
    )
    results = []
    for prompt, prediction in zip(prompts, predictions):
        for x in range(num_responses_per_prompt):
            if x >= len(prediction.choices):
                results.append("")
                continue
            results.append(
                prediction.choices[x]
                .message.content.replace("\n", "~| ")
                .replace("\t", " ")
            )
        
        # prompt = prompt.split("\t")
        # for x in range(num_responses_per_prompt):
        #     if x >= len(prediction.choices):
        #         prompt.append("")
        #         continue
        #     prompt.append(
        #         prediction.choices[x]
        #         .message.content.replace("\n", "~| ")
        #         .replace("\t", " ")
        #     )
        # results.append(tuple(prompt))
    return results


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--prompts",
        type=str,
        default="/home/shailyjb/pref_cult/corpus_collection/prompts_trial.tsv",
    )
    argparser.add_argument(
        "--output",
        type=str,
        default="/home/shailyjb/pref_cult/corpus_collection/responses_trial.tsv",
    )
    argparser.add_argument("--model", type=str, required=True)
    argparser.add_argument("--base_url", type=str, required=True)
    argparser.add_argument("--temperature", type=float, default=0.3)
    argparser.add_argument("--max_response_tokens", type=int, default=100)
    argparser.add_argument("--top_p", type=float, default=1.0)  # Don't alter this
    argparser.add_argument("--requests_per_minute", type=int, default=150)
    argparser.add_argument("--num_responses_per_prompt", type=int, default=1)
    args = argparser.parse_args()
    with open(args.prompts) as f:
        prompts = f.readlines()
    prompts = [prompt.strip() for prompt in prompts]
    header = prompts[0].split("\t")
    prompts = prompts[1:]  # skipping header for now

    client = AsyncOpenAI(api_key="EMPTY", base_url=args.base_url)
    results = vllm_inference(
        client,
        prompts,
        args.model,
        args.temperature,
        args.max_response_tokens,
        args.top_p,
        args.requests_per_minute,
        args.num_responses_per_prompt,
    )
    header = header.append("response")
    pd.DataFrame(results).to_csv(args.output, sep="\t", header=header, index=None)