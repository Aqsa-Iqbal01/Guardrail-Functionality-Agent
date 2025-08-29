from typing import Any
from openai import AsyncOpenAI
from agents import (
    Agent,
    GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    OpenAIChatCompletionsModel,
    TResponseInputItem,
    input_guardrail,
    output_guardrail,
    set_tracing_export_api_key,
    InputGuardrailTripwireTriggered,
)
from dotenv import find_dotenv, load_dotenv
import os
import asyncio

from pydantic import BaseModel

load_dotenv(find_dotenv(), override=True)

api_key = os.getenv("GEMINI_API_KEY")
base_url = os.getenv("GEMINI_BASE_URL")
model_name = os.getenv("GEMINI_MODEL_NAME")


client = AsyncOpenAI(api_key=api_key, base_url=base_url)
model = OpenAIChatCompletionsModel(openai_client=client, model=model_name)

set_tracing_export_api_key(api_key=api_key)


class MathOutPut(BaseModel):
    is_math: bool
    reason: str

class OutputScan(BaseModel):
    contains_political_content: bool
    reason: str


@input_guardrail
async def check_input(
    ctx: RunContextWrapper[Any], agent: Agent[Any], input_data: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    # print("input_data : ", input_data)

    input_agent = Agent(
        "InputGuardrailAgent",
        instructions="Check and verify if input is related to math",
        model=model,
        output_type=MathOutPut,
    )
    result = await Runner.run(input_agent, input_data, context=ctx.context)
    final_output = result.final_output
    # print(final_output)

    return GuardrailFunctionOutput(
        output_info=final_output, tripwire_triggered=not final_output.is_math
    )



@output_guardrail
async def check_output(
    ctx: RunContextWrapper[Any], agent: Agent[Any], output: str
) -> GuardrailFunctionOutput:
    moderation_agent = Agent(
        "OutputGuardrailAgent",
        instructions="Check if the output contains political content or references to political figures.",
        model=model,
        output_type=OutputScan,
    )
    result = await Runner.run(moderation_agent, output, context=ctx.context)
    scan_result = result.final_output

    return GuardrailFunctionOutput(
        output_info=scan_result,
        tripwire_triggered=scan_result.contains_political_content
    )


math_agent = Agent(
    "MathAgent",
    instructions="You are a math agent",
    model=model,
    input_guardrails=[check_input],
)

general_agent = Agent(
    "GeneralAgent",
    instructions="You are a helpful agent",
    model=model,
    output_guardrails=[check_output]
)


async def main():
    try:
    
        msg = input("Enter you question : ")
        result = await Runner.run(general_agent, msg)
        print(f"\n\n final output : {result.final_output}")
    except InputGuardrailTripwireTriggered:
        print("Error: Invalid prompt (Not math related).")
    except OutputGuardrailTripwireTriggered:
        print("Error: Output contains political content.")

asyncio.run(main())