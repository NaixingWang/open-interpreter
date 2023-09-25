import json

def convert_to_openai_messages(messages):
    new_messages = []

    for message in messages:  
        new_message = {
            "role": message["role"],
            "content": ""
        }

        if "message" in message:
            new_message["content"] = message["message"]

        if "code" in message:
            new_message["function_call"] = {
                "name": "execute",
                "arguments": json.dumps({
                    "language": message["language"],
                    "code": message["code"]
                }),
                # parsed_arguments isn't actually an OpenAI thing, it's an OI thing.
                # but it's soo useful! we use it to render messages to text_llms
                "parsed_arguments": {
                    "language": message["language"],
                    "code": message["code"]
                }
            }

        if "category" in message:
            new_message["function_call"] = {
                "name": "qa_dft",
                "arguments": json.dumps({
                    "question": message["question"],
                    "category": message["category"]
                }),
            }

        new_messages.append(new_message)

        if "output" in message:
            output = message["output"]

            new_messages.append({
                "role": "function",
                "name": "execute",
                "content": output
            })

        if "qa_answer" in message:
            qa_answer = message["qa_answer"]

            new_messages.append({
                "role": "function",
                "name": "qa_dft",
                "content": qa_answer
            })

    return new_messages