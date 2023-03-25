import openai
import logging
import asyncio

async def run_batch_whisper_api_async(files, batch_size = 64):
    length = len(files)
    transcripts = []
    for s in range(0, length, batch_size):
        futs = [
            asyncio.ensure_future(run_whisper_api_async(file)) 
            for file
            in files[s:s+batch_size]
        ]
        results = await asyncio.gather(*futs)
        transcripts.extend(results)
    return transcripts

async def run_whisper_api_async(file):
    fp = open(file, 'rb')
    try:
        whisper_result = await _execute_whisper_api_async(file=fp)
        fp.close()
    except openai.error.APIConnectionError as e:
        fp.close()
        logging.error(e)
        fp = open(file, 'rb')
        try:
            whisper_result = await _execute_whisper_api_async(file=fp)
        except openai.error.APIConnectionError as e:
            logging.error(e)
            whisper_result = None
        fp.close()
    return whisper_result

async def _execute_whisper_api_async(
        file,
        response_format='verbose_json',
        language='ko',
        model='whisper-1',
        request_timeout = 5,
        temperature = 0.2
):
    """Return result of whisper api.
       
       Params:
        file: file-like object (eg, file, BytesIO)
       
       Example of return value:
          <OpenAIObject> JSON {
              'duration': 6.71,
              'language': 'korean',
              'segments': [
                {
                    "avg_logprob": -0.338999931,
                    "compression_ration": 0.4215671,
                    "end": 3.2,
                    "id": 0,
                    "no_speech_prob": 0.02468,
                    "seek": 0,
                    "start": 0.0,
                    "temperature": 0.0,
                    "text": "마이크 테스트",
                    "tokens": [
                        1234,
                        5678,
                        12,
                        345,
                        67,
                        ...
                    ],
                    "transient": false
                },
                ....
              ]
          }
    """
    assert type(file) is not str
    assert openai.api_key is not None and openai.api_key != ''
    logging.info("START openai.Audio.atranscribe(...)")
    ret = await openai.Audio.atranscribe(
            model=model, 
            file=file, 
            response_format=response_format, 
            language=language,
            request_timeout = request_timeout,
            temperature = temperature
        ) 
    logging.info("OK openai.Audio.atranscribe(...)")
    logging.info(ret.text)
    return ret

"""
from .audio_processing import to_sliced_audio_segments
from typing import List
import io
import uuid
from pydub import AudioSegment
def run_sliced_whisper_api(
        audio_blob,
        slicing_ms=1000 * 60
):
    slicing_sec = slicing_ms / 1000
    audio_segments: List[AudioSegment] = to_sliced_audio_segments(
        audio_blob=audio_blob,
        slicing_ms=slicing_ms
    )

    # .filter( audio_duration >= 0.111 sec )
    audio_segments = [audio_segment for audio_segment in audio_segments if len(audio_segment) >= 111]

    results = []
    for i, audio_segment in enumerate(audio_segments):
        bo = io.BytesIO()
        audio_segment.export(bo, format="wav")
        bo.name = f"{uuid.uuid4()}.wav"

        whisper_result = _execute_whisper_api(bo)
        bo.close()

        results.extend([
            {
                "avg_logprob": res_segment.avg_logprob,
                "compression_ratio": res_segment.compression_ratio,
                "start": res_segment.start + (i * slicing_sec),
                "end": res_segment.end + (i * slicing_sec),
                "temperature": res_segment.temperature,
                "text": res_segment.text,
                "tokens": res_segment.tokens,
                "transient": res_segment.transient
            }
            for res_segment
            in whisper_result.segments
        ])

    return results
"""
