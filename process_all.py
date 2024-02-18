import pandas as pd
import jieba
from datetime import datetime
import datetime as dt
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from wordcloud import WordCloud, ImageColorGenerator
import os
from os import path
from imageio.v2 import imread
from matplotlib import pyplot as plt
from wordcloud import WordCloud,STOPWORDS

# 源csv文件的地址
ori_csv = r'聊天记录.csv'
# 下面3个分别是将源csv文件中Type为1的StrContent的内容提取出来的文件，第一个是双方共同的，第二个是0方的，第三个是1方的
out_txt = r"聊天记录.txt"
out_txt0 = r"聊天记录0.txt"
out_txt1 = r"聊天记录1.txt"
# 这下面这个是报错存储的文件地址，在这里可以看到整个过程的报错
err_file = "错误.txt"
# 这下面3个是词频统计后的输出文件地址，第一个是双方共同的，第二个是0方的，第三个是1方的
word_txt = r"聊天记录-词频统计.txt"
word_txt0 = r"聊天记录0-词频统计.txt"
word_txt1 = r"聊天记录1-词频统计.txt"
# 这下面3个是统计每天说话的字符数，第一个是双方共同的，第二个是0方的，第三个是1方的
word_day = r"聊天记录-具体的天.csv"
word_day0 = r"聊天记录0-具体的天.csv"
word_day1 = r"聊天记录1-具体的天.csv"
# 这下面3个是统计24h中说话的字符数，是所有天数的24h，第一个是双方共同的，第二个是0方的，第三个是1方的
word_hour = r"聊天记录-24h.csv"
word_hour0 = r"聊天记录0-24h.csv"
word_hour1 = r"聊天记录1-24h.csv"
# 下面这个是聊天所有字数的txt文件，分别包含总的和各自的
word_all = r"聊天记录-all.txt"
# 下面这个是设置热力图大小和颜色的
heatmap_r = 50
heatmap_l = 6
heatmap_dpi = 300
heatmap_color_l = '#ecd7dc'
heatmap_color_r = '#e7385c'
# 下面设置背景图的文件地址
back_path = r'背景图.png'
# 这下面是3个用于指定时间，比如，0-2点的每小时的字符数，第一个是双方共同的，第二个是0方的，第三个是1方的
word_hour_spe = r"聊天记录-spe-hour.csv"
word_hour_spe0 = r"聊天记录0-spe-hour.csv"
word_hour_spe1 = r"聊天记录1-spe-hour.csv"
# 这下面两个是设置指定时间的，0和1表示0-2点，最多能到23
hour_start = 0
hour_end = 1


# 这个函数负责将text中的内容写到报错文件里面
def write_err(text):
    with open(err_file, "a") as f:
        f.write(text + '\n')


# 这个函数负责生成out_txt，out_txt0与out_txt1
def out_txt_create():
    f = open(out_txt, "w", encoding="utf-8")
    f0 = open(out_txt0, "w", encoding="utf-8")
    f1 = open(out_txt1, "w", encoding="utf-8")
    data = pd.read_csv(ori_csv)
    # 这里只选择了Type为1的，要进行其他选择可以在下面这里改
    for index, row in data.iterrows():
        if row["Type"] == 1 and row["IsSender"] == 0:
            try:
                f0.write(str(row["StrContent"]) + '\n')
            except Exception as e:
                write_err(f'行: {index}, localId: {row["localId"]}, StrContent: {row["StrContent"]}, err: {e}')
        elif row["Type"] == 1 and row["IsSender"] == 1:
            try:
                f1.write(str(row["StrContent"]) + '\n')
            except Exception as e:
                write_err(f'行: {index}, localId: {row["localId"]}, StrContent: {row["StrContent"]}, err: {e}')
        if row["Type"] == 1:
            try:
                f.write(str(row["StrContent"]) + '\n')
            except Exception as e:
                write_err(f'行: {index}, localId: {row["localId"]}, StrContent: {row["StrContent"]}, err: {e}')
    f.close()
    f0.close()
    f1.close()


# 词频统计，in_path必须是out_txt_create函数生成的文件，负责统计文件中的词语的数量，然后输出前num个的词语到out_path中，并且会将前num个词频高的词return
def word_count(in_path, out_path, num):
    # 打开文本存储的txt文件
    text = open(in_path, "r", encoding="utf-8").read()
    # 输出词频统计的txt文件，可能需要改路径
    f_out = open(out_path, "w", encoding="utf-8")
    # jieba库的分词操作
    words = jieba.lcut(text)
    count = {}

    # 统计词频，这里没有统计一个字的
    for word in words:
        if len(word) == 1:
            continue
        else:
            # get的用法是在赋初值，仅第一次有效
            count[word] = count.get(word, 0) + 1

    # 词频排序
    items = list(count.items())
    items.sort(key=lambda x: x[1], reverse=True)

    # 输出前100，可以自己调
    for i in range(num):
        word, count = items[i]
        if out_path != 0:
            f_out.write("{0:<3}{1:>3}\n".format(word, count))
    if out_path != 0:
        f_out.close()
    return items


# 这是个样例函数，用于告诉你如何从源csv文件中获取信息
def get_obj():
    data = pd.read_csv(ori_csv)
    for index, row in data.iterrows():
        # 当前消息的长度
        l_num = len(str(row["StrContent"]))
        # 获取当前消息的时间戳
        t_num = time.localtime(row["CreateTime"])
        # 获取当前消息处于哪一个时刻，小时
        h_num = time.strftime("%H", t_num)
        # 获取当前消息处于哪一天
        d_str = time.strftime("%Y-%m-%d", t_num)


# 计算聊天开始到最后一共有几天，并输出start和end
def get_start_end():
    data = pd.read_csv(ori_csv)
    date_format = "%Y-%m-%d"
    start_time = datetime.utcfromtimestamp(int(data.iloc[0]["CreateTime"]))
    start_time = datetime.strftime(start_time, date_format)
    start_time = datetime.strptime(start_time, date_format)
    end_time = datetime.utcfromtimestamp(int(data.iloc[-1]["CreateTime"]))
    end_time = datetime.strftime(end_time, date_format)
    end_time = datetime.strptime(end_time, date_format)
    delta = end_time - start_time
    return delta.days + 1, start_time, end_time


# 统计每天的字符数，生成文件word_day, word_day0, word_day1, 并将统计结果return出去
def word_count_day():
    # 统计具体某天的字典
    count_d = {}
    count_d0 = {}
    count_d1 = {}
    data = pd.read_csv(ori_csv)
    for index, row in data.iterrows():
        # 当前消息的长度
        l_num = len(str(row["StrContent"]))
        # 获取当前消息的时间戳
        t_num = time.localtime(row["CreateTime"])
        # 获取当前消息处于哪一天
        d_str = time.strftime("%Y-%m-%d", t_num)
        # 是消息并且IsSender为0的情况
        if row["Type"] == 1 and row["IsSender"] == 0:
            count_d0[d_str] = count_d0.get(d_str, 0) + l_num
        # 是消息并且IsSender为1的情况
        elif row["Type"] == 1 and row["IsSender"] == 1:
            count_d1[d_str] = count_d1.get(d_str, 0) + l_num
        # 是消息并且总的情况
        if row["Type"] == 1:
            count_d[d_str] = count_d.get(d_str, 0) + l_num
    count_d = {key: [value] for key, value in count_d.items()}
    count_d0 = {key: [value] for key, value in count_d0.items()}
    count_d1 = {key: [value] for key, value in count_d1.items()}
    df = pd.DataFrame(count_d)
    df.to_csv(word_day, index=False)
    df = pd.DataFrame(count_d0)
    df.to_csv(word_day0, index=False)
    df = pd.DataFrame(count_d1)
    df.to_csv(word_day1, index=False)
    count_d = {key: value[0] for key, value in count_d.items()}
    count_d0 = {key: value[0] for key, value in count_d0.items()}
    count_d1 = {key: value[0] for key, value in count_d1.items()}
    return count_d, count_d0, count_d1


# 统计24h的字符数，生成文件word_hour, word_hour0, word_hour1, 并将统计结果return出去
def word_count_hour():
    data = pd.read_csv(ori_csv)
    # 统计24h的字典
    count = {}
    count0 = {}
    count1 = {}
    for i in range(24):
        obj = str(i).rjust(2, '0')
        count[obj] = count.get(obj, 0)
        count0[obj] = count0.get(obj, 0)
        count1[obj] = count1.get(obj, 0)
    for index, row in data.iterrows():
        # 当前消息的长度
        l_num = len(str(row["StrContent"]))
        # 获取当前消息的时间戳
        t_num = time.localtime(row["CreateTime"])
        # 获取当前消息处于哪一个时刻，小时
        h_num = time.strftime("%H", t_num)
        # 获取当前消息处于哪一天
        # 是消息并且IsSender为0的情况
        if row["Type"] == 1 and row["IsSender"] == 0:
            count0[h_num] = count0.get(h_num, 0) + l_num
        # 是消息并且IsSender为1的情况
        elif row["Type"] == 1 and row["IsSender"] == 1:
            count1[h_num] = count1.get(h_num, 0) + l_num
        # 是消息并且总的情况
        if row["Type"] == 1:
            count[h_num] = count.get(h_num, 0) + l_num
    count = {key: [value] for key, value in count.items()}
    count0 = {key: [value] for key, value in count0.items()}
    count1 = {key: [value] for key, value in count1.items()}
    df = pd.DataFrame(count)
    df.to_csv(word_hour, index=False)
    df = pd.DataFrame(count0)
    df.to_csv(word_hour0, index=False)
    df = pd.DataFrame(count1)
    df.to_csv(word_hour1, index=False)
    count = {key: value[0] for key, value in count.items()}
    count0 = {key: value[0] for key, value in count0.items()}
    count1 = {key: value[0] for key, value in count1.items()}
    return count, count0, count1


# 统计特定时间的字符数，生成文件word_hour, word_hour0, word_hour1, 并将统计结果return出去
def word_count_special_hour(start, end):
    data = pd.read_csv(ori_csv)
    # 统计24h的字典
    count = {}
    count0 = {}
    count1 = {}
    for i in range(24):
        obj = str(i).rjust(2, '0')
        count[obj] = count.get(obj, 0)
        count0[obj] = count0.get(obj, 0)
        count1[obj] = count1.get(obj, 0)
    for index, row in data.iterrows():
        # 当前消息的长度
        l_num = len(str(row["StrContent"]))
        # 获取当前消息的时间戳
        t_num = time.localtime(row["CreateTime"])
        # 获取当前消息处于哪一个时刻，小时
        h_num = time.strftime("%H", t_num)
        # 获取当前消息处于哪一天
        # 是消息并且IsSender为0的情况
        if row["Type"] == 1 and row["IsSender"] == 0 and start <= int(h_num) <= end:
            count0[h_num] = count0.get(h_num, 0) + l_num
        # 是消息并且IsSender为1的情况
        elif row["Type"] == 1 and row["IsSender"] == 1 and start <= int(h_num) <= end:
            count1[h_num] = count1.get(h_num, 0) + l_num
        # 是消息并且总的情况
        if row["Type"] == 1 and start <= int(h_num) <= end:
            count[h_num] = count.get(h_num, 0) + l_num
    count = {key: [value] for key, value in count.items()}
    count0 = {key: [value] for key, value in count0.items()}
    count1 = {key: [value] for key, value in count1.items()}
    df = pd.DataFrame(count)
    df.to_csv(word_hour_spe, index=False)
    df = pd.DataFrame(count0)
    df.to_csv(word_hour_spe0, index=False)
    df = pd.DataFrame(count1)
    df.to_csv(word_hour_spe1, index=False)
    count = {key: value[0] for key, value in count.items()}
    count0 = {key: value[0] for key, value in count0.items()}
    count1 = {key: value[0] for key, value in count1.items()}
    return count, count0, count1


# 统计总的字符数，生成文件word_all, 并将统计结果return出去
def word_count_all():
    data = pd.read_csv(ori_csv)
    # 统计字数的变量
    num = 0
    num0 = 0
    num1 = 0
    f = open(word_all, "w", encoding="utf-8")
    for index, row in data.iterrows():
        # 当前消息的长度
        l_num = len(str(row["StrContent"]))
        # 获取当前消息的时间戳
        t_num = time.localtime(row["CreateTime"])
        # 获取当前消息处于哪一个时刻，小时
        h_num = time.strftime("%H", t_num)
        # 获取当前消息处于哪一天
        d_str = time.strftime("%Y-%m-%d", t_num)
        # 是消息并且IsSender为0的情况
        if row["Type"] == 1 and row["IsSender"] == 0:
            num0 += l_num
        # 是消息并且IsSender为1的情况
        elif row["Type"] == 1 and row["IsSender"] == 1:
            num1 += l_num
        # 是消息并且总的情况
        if row["Type"] == 1:
            num += l_num
    # 保存数据到文件
    f.write(f"总聊天记录字数：{num}\nIsSender 0所发聊天记录字数：{num0}\nIsSender 1所发聊天记录字数：{num1}\n")
    f.close()


# 热力图所需函数，数据生成函数
def generate_data(start_time, day_num, text_path):
    l_table = []
    df = pd.read_csv(text_path)
    for i in range(day_num):
        now_time = start_time + dt.timedelta(days=i)
        if now_time.strftime("%Y-%m-%d") in df.columns:
            l_table.append(df.iloc[0][now_time.strftime("%Y-%m-%d")])
        else:
            l_table.append(0)
    data = np.array(l_table)
    dates = [start_time + dt.timedelta(days=i) for i in range(day_num)]
    return dates, data


# 热力图所需函数，数据封装函数
def calendar_array(dates, data):
    w_d = 0
    w_d_num = 0
    tt = []
    for d in dates:
        if w_d == 0:
            w_d = d.isocalendar()[1]
            w_d_num = w_d
        if w_d != d.isocalendar()[1]:
            w_d = d.isocalendar()[1]
            w_d_num += 1
        tt.append([w_d_num, d.isocalendar()[2]])
    i, j = zip(*tt)
    i = np.array(i) - min(i)
    j = np.array(j) - 1
    ni = max(i) + 1

    calendar = np.nan * np.zeros((ni, 7))
    calendar[i, j] = data
    return i, j, calendar


# 热力图所需函数，横轴标签（星期）函数
def label_days(ax, dates, i, j, calendar):
    ni, nj = calendar.shape
    day_of_month = np.nan * np.zeros((ni, 7))
    day_of_month[i, j] = [d.day for d in dates]

    for (i, j), day in np.ndenumerate(day_of_month):
        if np.isfinite(day):
            ax.text(j, i, int(day), ha='center', va='center')

    ax.set(xticks=np.arange(7),
           xticklabels=['M', 'T', 'W', 'R', 'F', 'S', 'S'])
    ax.xaxis.tick_top()


# 热力图所需函数，纵轴标签（月份）函数
def label_months(ax, dates, i, j, calendar, start_year):
    month_num = (dates[-1].year - dates[0].year) * 12 + dates[-1].month - dates[0].month + 1
    month_labels = np.array(
        [f'{dates[0].year + (i + dates[0].month - 1) // 12}-{str((i + dates[0].month - 1) % 12 + 1).rjust(2, "0")}' for
         i in range(month_num)])
    months = np.array([d.month + 12 * (d.year - start_year) for d in dates])
    # print(months)
    uniq_months = sorted(set(months))
    # print(uniq_months)
    yticks = []
    for m in uniq_months:
        mid_arr = i[months == m]
        if m == dates[0].month:
            mid_arr = mid_arr[mid_arr <= 10]
        yticks.append(mid_arr.mean())
    labels = [month_labels[m - dates[0].month] for m in uniq_months]
    ax.set(yticks=yticks)
    ax.set_yticklabels(labels, rotation=0)


# 热力图所需函数，传入日历数据和日期，输出日历图像
def calendar_heatmap(ax, dates, data, start_year):
    i, j, calendar = calendar_array(dates, data)
    newcmap = LinearSegmentedColormap.from_list('chaos', [heatmap_color_l, heatmap_color_r])
    im = ax.imshow(calendar, interpolation='none', cmap=newcmap)
    label_days(ax, dates, i, j, calendar)
    label_months(ax, dates, i, j, calendar, start_year)
    ax.figure.colorbar(im)


# 热力图生成，text_path是word_day, word_day0, word_day1其中一个
def heatmap_create(text_path):
    day_num, start_time, end_time = get_start_end()
    dates, data = generate_data(start_time, day_num, text_path)
    fig, ax = plt.subplots(figsize=(heatmap_l, heatmap_r))
    calendar_heatmap(ax, dates, data, int(datetime.strftime(start_time, "%Y")))
    ax.set_title(f"{datetime.strftime(start_time, '%Y')}-{datetime.strftime(end_time, '%Y')}")
    plt.savefig(f"热力图{datetime.strftime(start_time, '%Y')}-{datetime.strftime(end_time, '%Y')}.png", dpi=heatmap_dpi)
    plt.show()


# 词云图生成，text_path是out_txt, out_txt0, out_txt1其中一个
def wordcloud_create(text_path, num):
    words = word_count(text_path, "词频统计-{num}.txt", num)
    t_l = []
    # 输出前100，可以自己调
    for i in range(num):
        word, count = words[i]
        t_l.append(word)
    word_s = ' '.join(t_l)
    d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
    imgname2 = d + '/词云图_colored.jpg'
    back_coloring = imread(back_path)
    font_path = d + '/SourceHanSerifK-Light.otf'

    stopwords = set()
    content = [line.strip() for line in open('hit_stopwords.txt', 'r', encoding='utf-8').readlines()]
    stopwords.update(content)

    wc = WordCloud(font_path=font_path, background_color="white", max_words=2000, mask=back_coloring,
                   max_font_size=100, random_state=42, width=1000, height=860, margin=2, stopwords = stopwords)

    wc.generate(word_s)

    # create coloring from image
    image_colors_byImg = ImageColorGenerator(back_coloring)

    # show
    # we could also give color_func=image_colors directly in the constructor
    plt.imshow(wc.recolor(color_func=image_colors_byImg), interpolation="bilinear")
    plt.axis("off")
    # plt.figure()
    # plt.imshow(back_coloring, interpolation="bilinear")
    # plt.axis("off")
    plt.show()

    # save wordcloud
    wc.to_file(path.join(d, imgname2))


# 只需要在main函数中操作即可，你不需要的操作注释掉就可以了，这下面只是将所有要用的给弄出来，按着先后顺序弄的，比如你不需要生成热力图，就将生成热力图那行注释掉
def main():
    if os.path.exists(err_file):
        os.remove(err_file)
    # 首先只提取StrContent
    out_txt_create()
    # 进行分词
    word_count(out_txt, word_txt, 100)
    word_count(out_txt0, word_txt0, 100)
    word_count(out_txt1, word_txt1, 100)
    # 统计每天字符数
    word_count_day()
    # 统计24h字符数
    word_count_hour()
    # 统计特定时间的字符数，比如下面统计的是0-2点的字符数
    word_count_special_hour(hour_start, hour_end)
    # 统计总字符数
    word_count_all()
    # # 生成热力图
    # heatmap_create(word_day)
    # 生成词云图
    wordcloud_create(out_txt, 600)


if __name__ == '__main__':
    main()
