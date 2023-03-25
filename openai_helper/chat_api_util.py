import openai
import logging
import asyncio

async def run_batch_chat_completion_api_async(messages_list, batch_size = 64):
    length = len(messages_list)
    ret = []
    for s in range(0, length, batch_size):
        futs = [
            asyncio.ensure_future(run_chat_completion_api_async(messages)) 
            for messages
            in messages_list[s:s+batch_size]
        ]
        results = await asyncio.gather(*futs)
        ret.extend(results)
    return ret

async def run_chat_completion_api_async(messages):
    try:
        ret = await _execute_chat_completion_api_async(messages=messages)
    except openai.error.APIConnectionError as e:
        logging.error(e)
        try:
            ret = await _execute_chat_completion_api_async(messages=messages)
        except openai.error.APIConnectionError as e:
            logging.error(e)
            ret = None
    return ret 

async def _execute_chat_completion_api_async(
        messages,
        language='ko',
        model='gpt-3.5-turbo',
        # request_timeout = 5,
        temperature = 1,
        top_p = 1,
        n = 1,
        stream = False,
        presence_penalty = 0,
        frequency_penalty = 0,
        logit_bias = None,
        user='default-user'
):
    """Return result of chatCompletion api.
    """
    assert openai.api_key is not None and openai.api_key != ''
    logging.info("START openai.ChatCompletion.acreate(...)")
    ret = await openai.ChatCompletion.acreate(
            model=model, 
            # request_timeout = request_timeout,
            temperature = temperature,
            top_p = top_p,
            n = n,
            stream = stream,
            presence_penalty = presence_penalty,
            frequency_penalty = frequency_penalty,
            messages=messages,
            user=user
        ) 
    logging.info("OK openai.ChatCompletion.acreate(...)")
    logging.info(ret.choices[0].message.content)
    return ret


