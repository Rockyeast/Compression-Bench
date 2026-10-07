import sys, json, tiktoken
enc = tiktoken.get_encoding("cl100k_base")

if "--batch" in sys.argv:
    # Batch mode: read JSON array, return JSON array of token counts
    data = json.loads(sys.stdin.read())
    counts = [len(enc.encode(text)) for text in data]
    print(json.dumps(counts))
else:
    # Single mode: read stdin, return count
    print(len(enc.encode(sys.stdin.read())))
