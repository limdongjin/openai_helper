import logging
import hydra
import openai
import asyncio
from openai_helper.chat_api_util import run_batch_chat_completion_api_async

async def test_chat_completion():
    questions_list = [
        ['안녕 지피티'],
        ['오늘 날씨 정말 좋다',
        '코딩하기 좋은 날씨야'],
        ['음성인식과 자연어처리를 이용해서 챗봇을 만들고싶어',
        '어떤 것을 공부하면 좋을까?'],
        ['배고픈데 뭐먹을까?'],
        ['챗지피티 속도를 더 빠르게 쓰고싶어...', '더 빠른 프로그램을 짜고싶어'],
        ['현재 대한민국 부동산 시장에 대해 토론해보자. 먼저 너부터 의견을 말해봐'],
        ['오늘은 기분이 좋아', '왜냐면 날씨가 정말 좋거든', '일이 잘 풀릴거같아'],
        ['하이'],
        ['도커에 대해 쉽게 설명해줘'],
        ['카프카에 대해 설명해줘'],
        ['쿠버네티스에 대해 설명해줘'],
        ['멋쟁이사자처럼 알아?'],
        ['이두희 알아?']
    ]
    messages_list = [
        [
            {'role': 'user', 'content': question} 
            for question
            in questions
        ] 
        for questions 
        in questions_list
    ]
    res = await run_batch_chat_completion_api_async(messages_list)
    logging.info(res)

@hydra.main(version_base=None, config_path='../conf', config_name='config')
def main(cfg):
    openai_cfg = cfg.openai
    openai.api_key = openai_cfg.api_key

    asyncio.run(test_chat_completion())


if __name__ == '__main__':
    main()
