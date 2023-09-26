import litellm
from ..utils.merge_deltas import merge_deltas
from ..utils.parse_partial_json import parse_partial_json
from ..utils.convert_to_openai_messages import convert_to_openai_messages
import tokentrim as tt
import json


function_schema_code = {
  "name": "execute",
  "description":
  "Executes code on the user's machine, **in the users local environment**, and returns the output",
  "parameters": {
    "type": "object",
    "properties": {
      "language": {
        "type": "string",
        "description":
        "The programming language (required)",
        "enum": ["python", "R", "shell", "applescript", "javascript", "html"]
      },
      "code": {
        "type": "string",
        "description": "The code to execute (required)"
      }
    },
    "required": ["language", "code"]
  },
}
function_schema_qa = {
  "name": "qa_dft",
  "description":
  "Answer the profetional questions related to integrated circuit (IC) design-for-test (DFT),\
    Automatic Test Pattern Generation (ATPG), test compression, Logic Built-in-self-test (LBIST),\
    Memory Built-in-self-test (MBIST), Diagnosis, Silicon yield analysis and \
    Tessent (a series of electronic design automation (EDA) tools in IC testing).",
  "parameters": {
    "type": "object",
    "properties": {
      "category": {
        "type": "string",
        "description": "The category of the question (It should **ALWAYS** be the **FIRST** property given by you).) \
          'ATPG_General' means that the question is a general question in ATPG field, such as term explanation.\
          'Tessent_Commands' means that the question is about the usage of Tessent tool.\
          'Tessent_DRC' means that the questions is about the Design Rule Checking (DRC) in Tessent tool, i.e. DRC rule explation, analysis and fixing.\
          If you cannot determine which is the best match, put the question into 'ATPG_General' category.",
        "enum": ["ATPG_General", "Tessent_Commands", "Tessent_DRC"]
      },
      "question": {
        "type": "string",
        "description":
        "The summarized question related to IC testing. Should only contain infos related to IC testing.",
      },
      # This is a trick that allow we to know when the output of infos are finished.
      "finished": {
        "type": "string",
        "description":
        "Indicate whether the stream typing of other properties is finished (It should **ALWAYS** be the **LAST** property given by you).",
        "enum": ["yes", "no"]
      }
    },
    "required": ["category", "question", "finished"]
  },
}

def setup_openai_coding_llm(interpreter):
    """
    Takes an Interpreter (which includes a ton of LLM settings),
    returns a OI Coding LLM (a generator that takes OI messages and streams deltas with `message`, `language`, and `code`).
    """

    def coding_llm(messages):
        
        # Convert messages
        messages = convert_to_openai_messages(messages)

        # Add OpenAI's reccomended function message
        messages[0]["content"] += "\n\nOnly use the functions you have been provided with."
        messages[0]["content"] += "\n\nSpecial Notes: \
                                    For math questions, **never** trust yourself. You should always call proper package and use code to solve the math questions. \
                                    For questions related to integrated circuit (IC) testing, including design-for-test (DFT), Automatic Test Pattern Generation (ATPG), \
                                    test compression, Logic Built-in-self-test (LBIST), Memory Built-in-self-test (MBIST), Diagnosis, Silicon yield analysis and Tessent \
                                    (a series of electronic design automation (EDA) tools in IC testing area), **ALWAYS** using qa_dft function to get answer. \
                                    Normally, qa_dft gives you a clear answer. In case that the answer given by qa_dft is hard to understand. **ALWAYS** double check if the \
                                    answer is crystal clear.If you think the answer is not clear. **Rephrase the user question and ask again using qa_dft function.** \
                                    You are **NOT** a specialist in IC testing, do not try to answer the question by yourself. You can **ONLY** use qa_dft function to get answers."

        # Seperate out the system_message from messages
        # (We expect the first message to always be a system_message)
        system_message = messages[0]["content"]
        messages = messages[1:]

        # Trim messages, preserving the system_message
        messages = tt.trim(messages=messages, system_message=system_message, model=interpreter.model)

        if interpreter.debug_mode:
            print("Sending this to the OpenAI LLM:", messages)

        # Create LiteLLM generator
        params = {
            'model': interpreter.model,
            'messages': messages,
            'stream': True,
            'functions': [function_schema_code, function_schema_qa]
        }

        # Optional inputs
        if interpreter.api_base:
            params["api_base"] = interpreter.api_base
        if interpreter.api_key:
            params["api_key"] = interpreter.api_key
        if interpreter.max_tokens:
            params["max_tokens"] = interpreter.max_tokens
        if interpreter.max_retries:
            params["max_retries"] = interpreter.max_retries
        if interpreter.temperature:
            params["temperature"] = interpreter.temperature
        
        # These are set directly on LiteLLM
        if interpreter.max_budget:
            litellm.max_budget = interpreter.max_budget
        if interpreter.debug_mode:
            litellm.set_verbose = True

        response = litellm.completion(**params)

        accumulated_deltas = {}
        language = None
        code = ""
        # category = None
        # question = ""
        finished = False

        for chunk in response:

            if ('choices' not in chunk or len(chunk['choices']) == 0):
                # This happens sometimes
                continue

            delta = chunk["choices"][0]["delta"]

            # Accumulate deltas
            accumulated_deltas = merge_deltas(accumulated_deltas, delta)

            if "content" in delta and delta["content"]:
                yield {"message": delta["content"]}

            if ("function_call" in accumulated_deltas
                and "arguments" in accumulated_deltas["function_call"]):
                arguments = accumulated_deltas["function_call"]["arguments"]
                arguments = parse_partial_json(arguments)

                if not arguments:
                    continue

                func_name = accumulated_deltas["function_call"]["name"]

                if func_name == "qa_dft":
                    if not finished and "finished" in arguments:
                        finished = True
                        yield {"category": arguments["category"]}
                        yield {"question": arguments["question"]}
                        yield {"finished": "yes"}

                elif func_name == "execute":
                    if (language is None
                        and "language" in arguments
                        and "code" in arguments # <- This ensures we're *finished* typing language, as opposed to partially done
                        and arguments["language"]):
                        language = arguments["language"]
                        yield {"language": language}
                    
                    if language is not None and "code" in arguments:
                        # Calculate the delta (new characters only)
                        code_delta = arguments["code"][len(code):]
                        # Update the code
                        code = arguments["code"]
                        # Yield the delta
                        if code_delta:
                          yield {"code": code_delta}
                else:
                    # The following handles the functions hallucinated by LLM
                    # if interpreter.debug_mode:
                    # yield {"message": f"You have used a non-exist function named: {func_name}. You should avoid using it."}
                    # yield {"no_llm_message": f"\n\nThe LLM hullucinates with a new function named: {func_name}.\n\n"}
                    yield {"language": func_name}
                    # Calculate the delta (new characters only)
                    code_delta = arguments[len(code):]
                    # Update the code
                    code = arguments
                    # Yield the delta
                    if code_delta:
                      yield {"code": code_delta}
            
    return coding_llm