import interpreter
interpreter.auto_run = True
interpreter.model = "gpt-3.5-turbo"
interpreter.temperature = 0

def test_hello_world():
    interpreter.reset()
    messages = interpreter.chat("""Please reply with just the words "Hello, World!" and nothing else. Do not run code.""")
    assert messages == [{'role': 'user', 'message': 'Please reply with just the words "Hello, World!" and nothing else. Do not run code.'}, {'role': 'assistant', 'message': 'Hello, World!'}]

def test_math():
    interpreter.reset()
    messages = interpreter.chat("""Please perform the calculation 27073*7397 then reply with just the integer answer with no commas or anything, nothing else.""")
    assert "200258981" in messages[-1]["message"]

def test_delayed_exec():
    interpreter.reset()
    interpreter.chat("""Can you write a single block of code and execute it that prints something, then delays 1 second, then prints something else? No talk just code. Thanks!""")

def test_nested_loops_and_multiple_newlines():
    interpreter.reset()
    interpreter.chat("""Can you write a nested for loop in python and shell and run them? Also put 1-3 newlines between each line in the code. Thanks!""")

def test_markdown():
    interpreter.reset()
    interpreter.chat("""Hi, can you test out a bunch of markdown features? Try writing a fenced code block, a table, headers, everything. DO NOT write the markdown inside a markdown code block, just write it raw.""")
