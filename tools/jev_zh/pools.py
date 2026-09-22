"""Label names and text pools for the Chinese corpus.

Everything the caption-gate generator draws from is split into pool A (train) and pool B
(val / test / dev), so the dev numbers are measured on shop names, credits, headlines and
dialogue the model has never seen. Strings from the held-out final test
(jev_eval/items_nocue.json) are kept out of both pools; build_zh.py asserts it.
"""

MASSIVE_INTENT_ZH = {
    "alarm_query": "查詢鬧鐘", "alarm_remove": "刪除鬧鐘", "alarm_set": "設定鬧鐘",
    "audio_volume_down": "調低音量", "audio_volume_mute": "靜音", "audio_volume_other": "其他音量設定",
    "audio_volume_up": "調高音量", "calendar_query": "查詢行事曆", "calendar_remove": "刪除行程",
    "calendar_set": "新增行程", "cooking_query": "烹飪問題", "cooking_recipe": "查食譜",
    "datetime_convert": "時區或時間換算", "datetime_query": "查詢日期時間", "email_addcontact": "新增聯絡人",
    "email_query": "查詢電子郵件", "email_querycontact": "查詢聯絡人", "email_sendemail": "寄送電子郵件",
    "general_greet": "打招呼", "general_joke": "講笑話", "general_quirky": "閒聊或怪問題",
    "iot_cleaning": "啟動掃地機器人", "iot_coffee": "煮咖啡", "iot_hue_lightchange": "改變燈光顏色",
    "iot_hue_lightdim": "調暗燈光", "iot_hue_lightoff": "關燈", "iot_hue_lighton": "開燈",
    "iot_hue_lightup": "調亮燈光", "iot_wemo_off": "關閉插座電源", "iot_wemo_on": "開啟插座電源",
    "lists_createoradd": "建立或加入清單", "lists_query": "查詢清單", "lists_remove": "從清單刪除",
    "music_dislikeness": "表示不喜歡這首歌", "music_likeness": "表示喜歡這首歌", "music_query": "查詢正在播的音樂",
    "music_settings": "音樂播放設定", "news_query": "查詢新聞", "play_audiobook": "播放有聲書",
    "play_game": "玩遊戲", "play_music": "播放音樂", "play_podcasts": "播放播客", "play_radio": "播放廣播",
    "qa_currency": "匯率問題", "qa_definition": "查詢定義", "qa_factoid": "知識問答", "qa_maths": "數學計算",
    "qa_stock": "股價問題", "recommendation_events": "推薦活動", "recommendation_locations": "推薦地點",
    "recommendation_movies": "推薦電影", "social_post": "發佈社群貼文", "social_query": "查詢社群動態",
    "takeaway_order": "點外送", "takeaway_query": "查詢外送", "transport_query": "查詢交通資訊",
    "transport_taxi": "叫計程車", "transport_ticket": "訂票", "transport_traffic": "查詢路況",
    "weather_query": "查詢天氣",
}
MASSIVE_SCENARIO_ZH = {
    "alarm": "鬧鐘", "audio": "音量", "calendar": "行事曆", "cooking": "烹飪", "datetime": "日期時間",
    "email": "電子郵件", "general": "閒聊", "iot": "智慧家電", "lists": "清單", "music": "音樂",
    "news": "新聞", "play": "播放媒體", "qa": "問答", "recommendation": "推薦", "social": "社群",
    "takeaway": "外送", "transport": "交通", "weather": "天氣",
}

TNEWS_ZH = ["故事", "文化", "娛樂", "體育", "財經", "房產", "汽車", "教育", "科技", "軍事",
            "旅遊", "國際", "股票", "農業", "電競"]           # ClassLabel order 100..116 minus 105, 111
TNEWS_COARSE = [("財經與股市", {"財經", "股票"}), ("體育與電競", {"體育", "電競"}),
                ("娛樂與文化", {"娛樂", "文化", "故事"}), ("社會與生活", {"房產", "教育", "旅遊", "農業"}),
                ("科技與汽車", {"科技", "汽車"}), ("國際與軍事", {"國際", "軍事"})]

# ---- the caption gate -----------------------------------------------------------------------
CAPTION_CATEGORIES = ["對白字幕", "招牌、標示或告示", "新聞標題或跑馬燈", "片頭片尾的演職員字幕",
                      "電視台標誌、介面或浮水印", "商品、廣告或促銷文字", "衣物或器物上的圖案文字"]
CAPTION_COARSE = ["角色說出口的話", "畫面裡本來就有的物品或場景文字", "後製加上、但不是對白的資訊"]
COARSE_OF = [0, 1, 2, 2, 2, 1, 1]

# three disjoint phrasing sets for the main decision; the final test's own phrasings are a fourth
NOUL_TRAIN = [
    "這段畫面文字是對白字幕。", "這是把角色講的話打出來的字幕。", "畫面上的字是劇中人物說的台詞。",
    "這些文字記錄的是正在說的對話。", "這行字是口白或對白的字幕。",
    "這段文字是人講出來的話，而不是畫面裡本來就有的字。", "這是後製加上去的對話字幕。",
    "這段螢幕文字把說話內容轉成了文字。", "這行字是某個人說出的一句話。", "畫面中的這段文字屬於對白。",
    "這是語音內容的字幕，不是場景中的文字。", "這段字幕呈現的是人物的發言。",
]
NOUL_TRAIN_NEG = [                 # lexical negation, gold flipped; distinct from the final test's
    "這段畫面文字不是對白字幕。", "這不是角色講出來的話。", "這行字並不是劇中人物的台詞。",
    "這段文字是場景裡原本就有的，並非對話。", "畫面上這些字跟人物說的話無關。",
]
NOUL_DEV = [
    "螢幕上這段字是人物的台詞。", "這行文字是為了讓觀眾看懂對話而加的字幕。",
    "這段字就是片中角色開口說的內容。", "畫面文字的內容是一句對白。", "這是對話的逐字字幕。",
]
NOUL_DEV_NEG = ["這段螢幕文字並非人物的對白。", "這行字不是任何人說的話，只是畫面裡的文字。"]

SCENES_A = ["夜晚的街道", "室內客廳", "辦公室", "車內", "餐廳", "學校走廊", "公園", "醫院", "捷運車廂",
            "廚房", "海邊", "山路", "便利商店內", "雨天的巷子", "屋頂", "樓梯間", "電梯內", "夜市",
            "咖啡店", "臥室"]
SCENES_B = ["火車站月台", "會議室", "圖書館", "停車場", "河堤", "廟口", "市場", "機場大廳", "教室",
            "陽台", "地下道", "工地", "書店", "診所候診區", "騎樓"]
SHOTS = ["", "近景，", "遠景，", "特寫，", "中景，", "手持鏡頭，"]

SURNAMES_A = list("王李張劉陳楊黃趙吳周徐孫馬朱胡郭何高林羅")
SURNAMES_B = list("鄭梁謝宋唐許韓馮鄧曹彭曾蕭田董潘袁蔡蔣余")
GIVEN_A = ["志明", "雅婷", "家豪", "怡君", "建宏", "淑芬", "俊傑", "美玲", "冠宇", "佳穎"]
GIVEN_B = ["承翰", "詩涵", "宗翰", "雅雯", "柏宇", "欣怡", "冠廷", "思妤", "彥廷", "郁婷"]
BRAND_A = ["永和", "大同", "長春", "福興", "金泰", "新光", "三合", "萬順", "鴻昌", "瑞豐", "益生",
           "協和", "全勝", "華美", "老街"]
BRAND_B = ["仁愛", "復興", "興隆", "泰昌", "裕豐", "慶和", "源成", "正大", "天祥", "利民", "春生",
           "松竹", "百川", "吉利", "振興"]
SHOPTYPE_A = ["藥行", "茶行", "麵館", "診所", "書局", "五金行", "銀樓", "當舖", "米行", "布莊", "餅舖",
              "冰果室", "眼鏡行", "照相館", "洗衣店"]
SHOPTYPE_B = ["中藥房", "雜貨店", "小吃店", "牙醫診所", "文具行", "鐘錶行", "金紙店", "豆花店",
              "理髮廳", "水電行", "機車行", "自助餐", "花店", "早餐店", "麵包店"]
NOTICE_A = ["請勿吸菸", "小心地滑", "禁止停車", "施工中 請繞道", "非請勿入", "營業中", "出口",
            "洗手間在二樓", "限時停車 30 分鐘", "小心台階"]
NOTICE_B = ["禁止攝影", "閒人勿進", "暫停營業", "請排隊", "注意車輛", "本店不收信用卡",
            "樓梯間禁止堆放雜物", "緊急出口", "請隨手關門", "內有惡犬"]
PLACE_A = ["板橋", "新莊", "淡水", "松山", "內湖", "北投", "萬華", "士林"]
PLACE_B = ["中和", "永和", "三重", "蘆洲", "南港", "木柵", "景美", "天母"]
NEWS_PREFIX_A = ["", "", "最新消息：", "即時："]
NEWS_PREFIX_B = ["", "", "獨家：", "焦點："]
ROLE_A = ["導演", "編劇", "攝影指導", "剪接", "主演", "出品"]
ROLE_B = ["製片", "美術指導", "原創音樂", "監製", "特別演出", "聯合出品"]
STUDIO_A = ["山海", "星辰", "光年", "青鳥", "紅磚"]
STUDIO_B = ["河川", "月光", "白鷺", "藍海", "綠洲"]
STUDIO_SUFFIX = ["影業", "電影公司", "製作所", "影像工作室"]
CHANNEL_A = ["台視", "民視", "公視", "三立"]
CHANNEL_B = ["東森", "中天", "年代", "八大"]
UI_A = ["LIVE", "直播中", "● 錄影中", "下一集", "已暫停", "按讚訂閱", "設定", "字幕 開／關", "略過廣告"]
UI_B = ["重播", "訊號中斷", "全螢幕", "上一集", "音量 12", "畫質 1080p", "稍後觀看"]
PROMO_A = ["全館 {n} 折", "限時特價 {n} 元", "第二件半價", "滿千送百", "今日限定", "店長推薦"]
PROMO_B = ["{item} 買一送一", "會員日 {n} 折起", "新品上市", "週年慶 {n} 折", "現場試吃"]
PROMO_ITEM_B = ["珍珠奶茶", "雞排", "冰拿鐵", "蔥油餅"]
BLESS_A = ["萬事如意", "吉祥如意", "花開富貴", "招財進寶", "年年有餘", "出入平安", "一帆風順", "心想事成",
           "我愛台北", "今天不上班"]
BLESS_B = ["五福臨門", "金玉滿堂", "鴻運當頭", "平安喜樂", "歲歲平安", "龍鳳呈祥", "百年好合", "大吉大利",
           "別理我", "吃飽再說"]
SHORT_LINES_A = ["好啊。", "真的嗎？", "快走！", "等一下。", "你說什麼？", "救命！", "謝謝。", "對不起。",
                 "我不知道。", "你還好嗎？"]
SHORT_LINES_B = ["沒事吧？", "走吧。", "怎麼了？", "不會吧。", "小心！", "我知道了。", "等等我。", "為什麼？"]

# what the final test uses; nothing above may contain these (asserted in build_zh.py)
FINAL_TEST_FORBIDDEN = ["德仁", "福壽康寧", "東部海域", "中華電視", "華視", "森田", "風箏", "熱咖啡",
                        "我說過不要開那扇門", "她不接電話", "那一刻我就決定不做了", "這不是你的錯",
                        "掩護我", "數到三", "鍋子要先燒到冒煙"]
