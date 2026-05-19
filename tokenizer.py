import tiktoken
enc=tiktoken.encoding_for_model("gpt-4")


text="Hello I am rohit . I like trekking."


tokens=enc.encode(text)

print("Tokens", tokens)
print("Token count",len(tokens))



decode=enc.decode(tokens)

print("Decoded:",decode)
