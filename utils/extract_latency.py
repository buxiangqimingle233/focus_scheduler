import re

pattern = re.compile(r"Sum Exceeded Latency: (.*)")

ret = []
with open("nohup.out", "r") as f:
    for line in f:
        try:
            res = float(pattern.search(line).group(1))
            ret.append(res)
        except Exception:
            pass

odd, even = [], []
for i in range(0, len(ret), 2):
    odd.append(ret[i])
    even.append(ret[i + 1])

for i in odd:
    print(i)
print()
for j in even:
    print(j)
print(odd, even, sep='\n')