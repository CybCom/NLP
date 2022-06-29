from src.preprocess import read_excel

def grammer_filter(origin_str):
    def filterFunc(dicts):
        if "矿" in origin_str and dicts['机构名称'] != "自然资源":
            return False
        elif ("不动产" in origin_str or "房" in origin_str) and dicts['机构名称'] != "自然资源" and dicts['机构名称'] != "住建":
            return False
        elif "车" in origin_str and dicts['机构名称'] != "公安" and dicts['机构名称'] != "农业农村":
            return False
        elif "判决" in origin_str and dicts['机构名称'] != "公安" and dicts['机构名称'] != "法院":
            return False
        elif "海员" in origin_str and dicts['机构名称'] != "海事局":
            return False
        elif ("会计" in origin_str or "财会" in origin_str) and dicts['机构名称'] != "财政":
            return False
        elif "监护人" in origin_str and dicts['机构名称'] != "公安" and dicts['机构名称'] != "卫健委":
            return False
        elif "税" in origin_str and dicts['机构名称'] != "税务":
            return False
        elif "当事人" in origin_str and dicts['机构名称'] != "司法" and dicts['机构名称'] != "法院":
            return False
        elif "律师" in origin_str and dicts['机构名称'] != "司法":
            return False
        elif "侨" in origin_str and dicts['机构名称'] != "外办":
            return False
        elif "捐赠" in origin_str and dicts['机构名称'] != "红十字会" and dicts['机构名称'] != "民政":
            return False
        elif ("救助" in origin_str or "救养" in origin_str) and dicts['机构名称'] != "民政":
            return False
        elif "社保" in origin_str and dicts['机构名称'] != "人社":
            return False
        elif ("贷款" in origin_str or "公积金" in origin_str) and dicts['机构名称'] != "住建":
            return False
        elif "信访" in origin_str and dicts['机构名称'] != "信访局":
            return False
        elif ("电" in origin_str and dicts['机构名称'] != "电力公司") or ("林" in origin_str and dicts['机构名称'] != "林业局"):
            return False
        return True
    filename = "test_data.xlsx"
    sheet_name = "关联关系"
    lst = read_excel(filename, sheet_name)
    new_lst = list(filter(filterFunc, lst))
    return new_lst

