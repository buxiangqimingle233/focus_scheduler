f = open("gpt2-xl.yaml", "w")

print("gpt2-xl:", file=f)
for i in range(1, 193):
    print(f"- gpt2-xl_layer{i}: 8", file=f)