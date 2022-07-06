f = open("gpt2.yaml", "w")

print("gpt2:", file=f)
for i in range(1, 49):
    print(f"- gpt2_layer{i}: 16", file=f)