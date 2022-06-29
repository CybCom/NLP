from src.filter import grammer_filter
from src.sentence2vector import vector_similarity

if __name__ == '__main__':
    origin_str = input("需要匹配的数据元 = ")
    while True:
        match_pattern = input("pattren = ")  # 有single 和 combination
        if match_pattern != "single" and match_pattern != "combination":
            print("模式不对，请重新输入！")
            continue
        break
    topk = int(input("topk = "))
    lst = grammer_filter(origin_str)
    result_list = []
    for item in lst:
        if match_pattern == "single":
            tmp_list = [item, vector_similarity(item['字段中文（可忽略）'], origin_str)]
        else:
            s = str(item['机构名称'] if item['机构名称'] is not None else "    ") + " " + str(item['表中文名']) + " " + str(item['字段中文（可忽略）'])
            print(s)
            tmp_list = [item, vector_similarity(s, origin_str)]
        result_list.append(tmp_list)
    result_list.sort(key=lambda x: x[1], reverse=True)
    for i in range(0, topk):
        print(result_list[i][0]['机构名称'], result_list[i][0]['表中文名'], result_list[i][0]['字段中文（可忽略）'], result_list[i][1])



