import chromedriver_autoinstaller as chromedriver
import pandas as pd

chromedriver.install()
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from datetime import datetime

# 粉笔官网地址
url = 'https://fenbi.com/page/home'

# 账户信息
# 可以作为参数传递 TODO
username = '13083412325'
password = 'llh19920402'

# 配置Selenium Chrome WebDriver
driver = webdriver.Chrome()
wait = WebDriverWait(driver, 10)
# 使用Selenium打开URL
driver.get(url)


# 伪造登陆场景
def fake_login():
    # 点击header-content-logon类名的按钮
    # 程序员写css类名字时写错了
    login_button = driver.find_element_by_class_name('header-content-logon')
    login_button.click()

    # 等待弹出框出现，并点击使用用户名密码登录
    # 可以选择使用验证码登陆，但是更适合使用账号密码登陆
    switch_to_password_login = wait.until(
        EC.element_to_be_clickable(
            (By.XPATH, '/html/body/app-root/div[1]/fb-header/div[2]/div/div[2]/div[5]/div/span')))
    switch_to_password_login.click()

    # 勾选服务条款
    click_agreement = driver.find_element_by_xpath(
        '/html/body/app-root/div[1]/fb-header/div[2]/div/div[2]/div[6]/div[1]')
    click_agreement.click()

    # 输入用户名和密码
    username_input = driver.find_element_by_xpath('/html/body/app-root/div[1]/fb-header/div[2]/div/div[2]/div[1]/input')
    username_input.send_keys(username)
    password_input = driver.find_element_by_xpath('/html/body/app-root/div[1]/fb-header/div[2]/div/div[2]/div[2]/input')
    password_input.send_keys(password)

    # 点击登录按钮
    login_submit = driver.find_element_by_xpath('/html/body/app-root/div[1]/fb-header/div[2]/div/div[2]/div[4]/div')
    login_submit.click()


fake_login()

# 等待一段时间以便页面重定向完成
time.sleep(2)
print('已等待2秒')

# 登录完成后重定向到新页面
# 基于题库类别的不同存在不同的页面地址
# 可以作为参数传递 TODO
new_url = 'https://www.fenbi.com/spa/tiku/guide/catalog/sydw?prefix=syzc'
driver.get(new_url)


# 标准化题目配置
def normalize_config():
    # 点击自定义刷题选择刷题模式，等待弹框出现：选择背题模式，选择40道题并保存
    customization_mode_button = wait.until(EC.element_to_be_clickable((By.XPATH,
                                                                       '/html/body/app-root/div/app-main/div[2]/app-catalog/div/main/div[1]/div[2]/app-keypoint-catalog/div/div[1]/a')))
    # 弹出modal
    customization_mode_button.click()
    # 选中背题模式
    hard_mode_button = wait.until(
        EC.element_to_be_clickable((By.XPATH, '/html/body/app-customize-question/div/div/main/ul[1]/li[2]/a')))
    hard_mode_button.click()
    # 选中每轮40道题
    max_questions_button = wait.until(
        EC.element_to_be_clickable((By.XPATH, '/html/body/app-customize-question/div/div/main/ul[2]/li[8]/a')))
    max_questions_button.click()
    # 保存配置
    config_save_button = wait.until(
        EC.element_to_be_clickable((By.XPATH, '/html/body/app-customize-question/div/div/footer/button[2]')))
    config_save_button.click()


normalize_config()


def save_exercise_in_list(i, j, k, m, label_content):
    # 点击后页面会跳转到具体的题目页
    # 在没有作答的时候，题目页只会存在题目和选项
    # 但是当你在背题模式下，选择答案会自动弹出正确解和解答
    # 需要将全部的内容都触发弹出
    # 以您需要的方式查找元素，例如通过XPath
    # 需要隐藏底部的toolbar
    card_button = driver.find_element_by_css_selector('.fb-collpase-header.bg-color-gray-mid.border-gray-mid')
    card_button.click()
    # 所有的选项卡
    choices = driver.find_elements_by_class_name('options')
    tempLength = 0
    # 将所有的选项都进行选择，确保题解和正确答案被正确唤醒
    while len(choices) != tempLength:
        for choice in choices:
            # 选题的对错在此时不重要
            time.sleep(0.2)
            try:
                first_li_element = choice.find_element_by_xpath('./li')
                # 点击找到的li元素
                first_li_element.click()
                driver.execute_script("window.scrollBy(0, 1800);")
                tempLength = len(driver.find_elements_by_class_name('correct-text'))
            except Exception:
                # 出现异常就将文件暂存到时间戳文件中
                current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                # 设置文件名
                temp_file_name = f'temp-{current_time}.csv'
                # 使用字典创建一个DataFrame: 包括 题干 选项 正确答案 题解
                # 在讲全部的内容都抓取下来之前不进行 dataframe 封装
                temp_df = pd.DataFrame(
                    {'context': questions_list, 'choice': choices_list, 'correct_answer': correct_answers_list,
                     'question_answer': question_solution_list})
                print(temp_df[:10])

                # 将DataFrame导出为CSV文件
                temp_df.to_csv(temp_file_name, index=False)

    # 获取所有article标签
    questions = driver.find_elements_by_class_name("question-content")
    # 遍历问题，将问题存储为string 类型的list
    for question in questions:
        context = question.text.strip()
        questions_list.append(context)
        label_list.append(label_content)
    # 遍历问题选项，将问题选项存储为 list 类型的 list
    choices = driver.find_elements_by_class_name('options')
    for choice in choices:
        # 解析HTML代码
        html = choice.get_attribute('outerHTML')
        soup = BeautifulSoup(html, 'html.parser')
        # 从解析后的HTML中提取所有li元素
        li_elements = soup.find_all('li')
        # 将li元素存储为数组
        li_array = [li.text.strip() for li in li_elements]
        choices_list.append(li_array)

    # 遍历问题的正确答案，将问题选项存储为 string 类型的 list
    correct_answers = driver.find_elements_by_class_name('correct-text')
    print('correct-text length', len(correct_answers))
    for correct_answer in correct_answers:
        context = correct_answer.text.strip()
        correct_answers_list.append(context)

    # 遍历问题的题解，将问题的题解存储为 string 类型的 list
    # 题解存在两部分，其中第一部分是题解，第二部分是题目出处，目前仅使用第一部分，第二部分的用途尚不明晰
    question_solutions = driver.find_elements_by_tag_name('fb-ng-solution-detail-content')
    print('question_solutions length', len(question_solutions))
    temp_index = 0
    for question_solution in question_solutions:
        context = question_solution.text.strip()
        if temp_index % 2 == 0:
            question_solution_list.append(context)
        temp_index += 1

    print(len(questions_list))
    print(len(choices_list))
    print(len(correct_answers_list))
    print(len(question_solution_list))

    # 全部内容都爬取后，点击交卷
    submit_exercise_button = driver.find_element_by_class_name('submit-exercise')
    submit_exercise_button.click()
    # 点击确认
    confirm_submit_button = wait.until(
        EC.element_to_be_clickable((By.XPATH, '/html/body/fb-modal-dialog/div/div/div[2]/button[2]')))
    confirm_submit_button.click()
    # 交卷成功后会自动跳转到阅卷页面，这个页面没有什么有价值的信息，点击退出退回到题库页面
    quit_button = wait.until(EC.element_to_be_clickable((By.XPATH,
                                                         '/html/body/app-root/div/app-main/div[2]/app-solution/div/header/app-simple-nav-header/header/div/button')))
    quit_button.click()
    # 等待一段时间以便页面更新
    time.sleep(1)
    # find_correct_path(i, j, k, m)


def find_correct_path(i, j, k, m):
    seu = driver.find_element_by_class_name('key-points')
    sel = seu.find_elements_by_tag_name('li')
    se1 = sel[i]
    se1.click()
    driver.implicitly_wait(0.5)
    s1l = se1.find_elements_by_tag_name('li')
    ss1 = s1l[j]
    ss1.click()
    driver.implicitly_wait(0.5)
    s2l = ss1.find_elements_by_tag_name('point-tree')
    ss2 = s2l[k]
    if m >= 0:
        ss2.click()
        driver.implicitly_wait(0.5)
        s3l = ss2.find_elements_by_tag_name('point-tree')
        ss3 = s3l[m]
        ss3.find_elements_by_tag_name('div')[1].click()
    else:
        ss2.find_elements_by_tag_name('div')[1].click()


# 待优化逻辑
# 选中专项练习并获取所有专项细分
special_exercise_ul = driver.find_element_by_class_name('key-points')
special_exercise_list = special_exercise_ul.find_elements_by_tag_name('li')
label_list = []
questions_list = []
choices_list = []
correct_answers_list = []
question_solution_list = []
for i, single_exercise in enumerate(special_exercise_list):
    # 获取一级标签
    first_class_label = single_exercise.text.strip()
    # 基于页面逻辑一级标签必然可以打开展示二级标签
    single_exercise.click()
    # 找到全部的二级标签
    section1_list = single_exercise.find_elements_by_tag_name('li')
    for j, section1 in enumerate(section1_list):
        second_class_label = ''
        if j > 0:
            # 找到li标签下的p标签，其中包含二级题目类型
            section1_description_list = section1.find_elements_by_tag_name('p')
            second_class_label += section1_description_list[0].text.strip()
            # 打开二级标签的三级标签
            section1.click()
            driver.implicitly_wait(1)
            # 找到三级题型
            section2_list = section1.find_elements_by_tag_name('point-tree')
            for k, section2 in enumerate(section2_list):
                third_class_label = ''
                # 基于li标签下的div标签中是否内嵌span标签确认是否是最底层
                span_elements = section2.find_elements_by_tag_name('div')[0].find_elements_by_tag_name('span')
                section2_description_list = section2.find_elements_by_tag_name('p')
                third_class_label += section2_description_list[0].text.strip()
                # 存在span的情况下，说明存在四级细分
                if len(span_elements) > 0:
                    print('span exist')
                    section2.click()
                    driver.implicitly_wait(1)
                    section3_list = section2.find_elements_by_tag_name('point-tree')
                    for m, section3 in enumerate(section3_list):
                        fourth_class_label = ''
                        section3_description_list = section3.find_elements_by_tag_name('p')
                        fourth_class_label += section3_description_list[0].text.strip()
                        fourth_class_exercise_total = section3_description_list[1].text.strip().split('/')[1]
                        print(fourth_class_exercise_total)
                        if int(fourth_class_exercise_total) > 0:
                            # file_name = first_class_label + second_class_label + third_class_label + fourth_class_label + '.csv'
                            # 填充 label_pair
                            four_label_pair = {'first_label': first_class_label, 'second_label': second_class_label,
                                               'third_label': third_class_label, 'fourth_label': fourth_class_label}
                            # 找到当前这个li下的a标签点击之就可以跳转到做题页面, 前提是存在题目
                            fourth_class_exercise_div = section3.find_elements_by_tag_name('div')[1]
                            fourth_class_exercise_div.click()
                            # 等待一段时间以便页面更新
                            time.sleep(1)
                            print('已等待1秒')
                            tempTotal = 0
                            index = 1
                            while tempTotal < int(fourth_class_exercise_total):
                                save_exercise_in_list(i, j, k, m, four_label_pair)
                                tempTotal = index * 40
                                index += 1
                                if tempTotal < int(fourth_class_exercise_total):
                                    find_correct_path(i, j, k, m)

                else:
                    # 填充 label_pair
                    three_label_pair = {'first_label': first_class_label, 'second_label': second_class_label,
                                        'third_label': third_class_label, 'fourth_label': 'none'}
                    # span不存在的情况下，可以直接获取三级标签的题目总量
                    third_class_exercise_total = section2_description_list[1].text.strip().split('/')[1]
                    print(third_class_exercise_total)
                    if int(third_class_exercise_total) > 0:
                        # file_name = first_class_label + second_class_label + third_class_label + '.csv'
                        # 找到当前这个li下的a标签点击之就可以跳转到做题页面, 前提是存在题目
                        third_class_exercise_div = section2.find_elements_by_tag_name('div')[1]
                        third_class_exercise_div.click()
                        # 等待一段时间以便页面更新
                        time.sleep(1)
                        print('已等待1秒')
                        tempTotal = 0
                        index = 1
                        while tempTotal < int(third_class_exercise_total):
                            save_exercise_in_list(i, j, k, -1, three_label_pair)
                            tempTotal = index * 40
                            index += 1
                            if tempTotal < int(third_class_exercise_total):
                                find_correct_path(i, j, k, -1)

df = pd.DataFrame(
    {'context': questions_list, 'choice': choices_list, 'correct_answer': correct_answers_list,
     'question_answer': question_solution_list, 'label': label_list})
print(df[:10])

# 将DataFrame导出为CSV文件
df.to_csv('事业单位联考C-职测.csv', index=False)
