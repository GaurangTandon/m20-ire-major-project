from selenium.webdriver.firefox.options import Options
import bs4
import sys
import pandas as pd
import re
from random import random
from selenium import webdriver
from nltk.tokenize import sent_tokenize

opts = Options()
opts.headless = True
assert opts.headless


def get_text(element):
    full_text = ""

    for child in element.children:
        name = child.name
        if name in ["div", "img", "strong", "br", "code", "pre", "verbatim"]:
            continue

        newline = "" if name in ["ol", "li"] else "\n"
        text = ""
        try:
            child.get_text()
            text = get_text(child)
        except AttributeError:
            text = child

        full_text += newline + text

    return full_text


def replacer(x):
    return 'eg' if x.group(0) == 'eg' else 'example'


def main(url: str):
    browser = webdriver.Firefox(options=opts)
    browser.get(url)
    elm = browser.find_element_by_class_name('entry-content')
    html: str = elm.get_property("innerHTML")
    with open("debug", "w") as f:
        f.write(html)

    # with open("debug") as f:
    #     html = f.read()

    parser = bs4.BeautifulSoup(html, features="html.parser")

    full_text = get_text(parser)

    # remove footer
    footer_text_reg = re.compile(r"This article is contributed|Recommended posts", re.IGNORECASE)
    match = re.search(footer_text_reg, full_text)
    if match:
        idx = match.start()
        full_text = full_text[:idx]

    # collapse newlines
    newline_reg = re.compile(r"\n+")
    full_text = re.sub(newline_reg, "", full_text)

    # remove extra full stops
    full_text = re.sub(r"e\.?g\.?", replacer, full_text)

    full_text = re.sub(re.compile(r"Attention Reader!.*|Please write comments.*|Don.?t stop learning now.*", re.I), "",
                       full_text)

    # get sentences, make csv
    sentences = sent_tokenize(full_text)
    count = len(sentences)
    df = pd.DataFrame(list(zip(sentences, [0 for _ in range(count)])))

    hash = random()
    filename = "data-" + str(hash)[2:] + ".csv"
    df.to_csv(filename, index=False, header=False)
    print(f"Saved to {filename}")

    browser.quit()


if __name__ == '__main__':
    try:
        url = sys.argv[1]
        assert url
    except Exception:
        print("Usage: python3 main.py <URL>")
        exit(1)

    main(url)
