

def add_results2_to_results1(res1, res2):
    """Note: res2 results will overwrite res1 results if there is overlap"""
    for k, v in res2.items():
        if k == "method_dicts":
            for el in res2[k]:
                replaced = False
                for i, el2 in enumerate(res1[k]):
                    if el["name"] == el2["name"]:
                        res1[k][i] = el
                        replaced = True
                        break
                if not replaced:
                    res1[k].append(el)

        if isinstance(v, dict):
            for kk, vv in v.items():
                res1[k][kk] = vv
            print(k, res1[k].keys())

    return res1
