# prompts.py

# 首頁初始提示
INITIAL_PROMPT = "Hi 我是小紋，請先從下面三種刺青風格中選一個開始："

# 三種風格示意
INITIAL_STYLES = {
    "S1": {"text": "傳統刺青風格", "image": "/static/image/japense.jpeg"},
    "S2": {"text": "水彩刺青風格", "image": "/static/image/watercolor.jpeg"},
    "S3": {"text": "寫實刺青風格", "image": "/static/image/real.jpeg"},
}

# 6 個問題的設定
QUESTIONS_OPTIONS = [
    {
        "key": "Q1",
        "text": "1/6：請描述你想要的主題或元素：",
        "type": "input"
    },
    {
        "key": "Q2",
        "text": "2/6：請選擇你偏好的線條風格：",
        "type": "choice",
        "options": {
            "Q2A": "粗獷的線條",
            "Q2B": "細膩的線條"
        }
    },
    {
        "key": "Q3",
        "text": "3/6：請選擇你偏好的構圖方式：",
        "type": "choice",
        "options": {
            "Q3A": "對稱構圖",
            "Q3B": "不對稱構圖"
        }
    },
    {
        "key": "Q4",
        "text": "4/6：請選擇刺青位置與尺寸：",
        "type": "choice",
        "options": {
            "Q4A": "手臂外側 約10×10cm",
            "Q4B": "背部 中等尺寸",
            "Q4C": "小腿側面 約15×15cm"
        }
    },
    {
        "key": "Q5",
        "text": "5/6：請選擇風格或配色細節：",
        "type": "choice",
        "options": {
            "Q5A": "黑白線條",
            "Q5B": "鮮豔水彩",
            "Q5C": "細緻點綴"
        }
    },
    {
        "key": "Q6",
        "text": "6/6：請簡短描述你希望看到刺青時的心情：",
        "type": "input"
    }
]