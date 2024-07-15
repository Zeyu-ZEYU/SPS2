import copy


def draw_cdf(values, labels=None, num_slices=100):
    cdf_values = []
    values = copy.deepcopy(values)
    if isinstance(values[0], int) or isinstance(values[0], float):
        values = [values]
    maxs = []
    mins = []
    for vlst in values:
        vlst.sort()
        maxs.append(vlst[-1])
        mins.append(vlst[0])
    maxs.sort()
    mins.sort()
    max = maxs[-1]
    min = mins[0]
    interval = max - min
    len_slice = interval / num_slices
    for vlst in values:
        len_vlst = len(vlst)
        vlst_idx = 0
        temp_arr = []
        for i in range(num_slices):
            border = min + i * len_slice
            while True:
                if vlst_idx == len_vlst:
                    temp_arr.append(1)
                    break
                v = vlst[vlst_idx]
                if v < border:
                    vlst_idx += 1
                elif v == border:
                    temp_arr.append((vlst_idx + 1) / len_vlst)
                    vlst_idx += 1
                    break
                else:
                    temp_arr.append(vlst_idx / len_vlst)
                    break
        temp_arr.append(1)
        cdf_values.append(temp_arr)
    csv_str = ""
    for i in range(num_slices):
        border = min + i * len_slice
        csv_str += f",{border}"
    csv_str += f",{max}\n"
    if labels is None:
        labels = [i for i in range(len(values))]
    for i in range(len(cdf_values)):
        label = labels[i]
        cdf_lst = cdf_values[i]
        csv_str += f"{label}"
        for v in cdf_lst:
            csv_str += f",{v}"
        csv_str += "\n"
    return csv_str
