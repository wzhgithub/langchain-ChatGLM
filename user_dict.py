import os

with open("user_dict.txt", 'w') as wf:
    res = {}
    with open("name.txt", 'r') as nf:
        for line in nf:
            ls = line.strip('\n').split('__')
            for e in ls[1:-1]:
                if not e.isdigit():
                    v = res.get(e, 0)
                    res[e] = v+1
    for k, v in res.items():
        l = f"{k} {v}\n"
        wf.write(l)
    wf.flush()
