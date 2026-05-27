"""Shared normalization helpers for GUI skill storage identifiers."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from opengui.skills.data import Skill

GUI_SKILLS_DIRNAME = "gui_skills"

# ---------------------------------------------------------------------------
# Package name <-> display name mapping for Android
# ---------------------------------------------------------------------------

_ANDROID_PACKAGE_DISPLAY_NAMES: dict[str, str] = {
    # Social & Communication
    "com.tencent.mm": "微信/WeChat",
    "com.tencent.mobileqq": "QQ",
    "com.sina.weibo": "微博/Weibo",
    "com.zhihu.android": "知乎/Zhihu",
    "com.xingin.xhs": "小红书/RedNote",
    "com.twitter.android": "X/Twitter",
    "com.whatsapp": "WhatsApp",
    "org.telegram.messenger": "Telegram",
    "com.facebook.katana": "Facebook",
    "tv.danmaku.bili": "哔哩哔哩/Bilibili",
    # Shopping & Food
    "com.taobao.taobao": "淘宝/Taobao",
    "com.jingdong.app.mall": "京东/JD",
    "com.xunmeng.pinduoduo": "拼多多/Pinduoduo",
    "com.taobao.idlefish": "闲鱼/Xianyu",
    "com.sankuai.meituan": "美团/Meituan",
    "com.dianping.v1": "大众点评/Dianping",
    "com.pupumall.customer": "朴朴超市/PuPu",
    "cn.walmart.app": "沃尔玛/Walmart",
    "com.lucky.luckyclient": "瑞幸咖啡/Luckin",
    "com.yek.android.kfc.activitys": "肯德基/KFC",
    # Transport & Maps
    "com.sdu.didi.psnger": "滴滴出行/DiDi",
    "com.autonavi.minimap": "高德地图/Amap",
    "cn.caocaokeji.user": "曹操出行/CaoCao",
    "com.lalamove.huolala.client": "货拉拉/Huolala",
    "com.jingyao.easybike": "哈啰/Hellobike",
    "com.ygkj.chelaile.standard": "车来了/Chelaile",
    # Travel
    "ctrip.android.view": "携程/Ctrip",
    "cn.damai": "大麦/Damai",
    "com.csair.mbp": "南方航空/CSAir",
    "com.rytong.ceair": "东方航空/CEAir",
    "com.umetrip.android.msky.app": "航旅纵横/Umetrip",
    "com.MobileTicket": "12306",
    # Finance
    "com.eg.android.AlipayGphone": "支付宝/Alipay",
    "com.icbc": "工商银行/ICBC",
    "com.icbc.elife": "工银e生活",
    "com.unionpay": "云闪付/UnionPay",
    "com.finshell.wallet": "数字人民币/e-CNY",
    "com.pingan.paces.ccms": "平安口袋银行/PingAn",
    "com.chinamworld.bocmbci": "中国银行/BOC",
    "com.bochk.app.aos": "中银香港/BOCHK",
    "com.android.bankabc": "农业银行/ABC",
    "cn.gov.tax.its": "个人所得税/ITS",
    "com.usmart.stock": "uSMART HK",
    "com.usmart.sg.stock": "uSMART SG",
    # Entertainment
    "com.ss.android.ugc.aweme": "抖音/Douyin",
    "com.netease.cloudmusic": "网易云音乐/NetEase Music",
    "com.google.android.youtube": "YouTube",
    "com.bytedance.dreamina": "即梦/Dreamina",
    # Work & Productivity
    "com.ss.android.lark": "飞书/Lark",
    "com.tencent.wework": "企业微信/WeCom",
    "com.tencent.wemeet.app": "腾讯会议/VooV",
    "com.tencent.docs": "腾讯文档/Tencent Docs",
    "com.tencent.androidqqmail": "QQ邮箱/QQ Mail",
    "cn.wps.moffice_eng": "WPS Office",
    "com.microsoft.office.outlook": "Outlook",
    "com.microsoft.skydrive": "OneDrive",
    "com.microsoft.office.officehub": "Microsoft Office",
    "notion.id": "Notion",
    "md.obsidian": "Obsidian",
    # AI
    "com.deepseek.chat": "DeepSeek",
    "com.openai.chatgpt": "ChatGPT",
    "com.aliyun.tongyi": "通义千问/Tongyi",
    "ai.x.grok": "Grok",
    "com.google.android.apps.bard": "Gemini",
    "com.tencent.hunyuan.app.chat": "腾讯混元/Hunyuan",
    "com.pocketpalai": "PocketPal AI",
    # Reading & Cloud
    "com.tencent.weread": "微信读书/WeRead",
    "com.baidu.netdisk": "百度网盘/Baidu Netdisk",
    "net.csdn.csdnplus": "CSDN",
    # Google
    "com.android.chrome": "Chrome",
    "com.google.android.gm": "Gmail",
    "com.google.android.apps.maps": "Google Maps",
    "com.google.android.apps.photos": "Google Photos",
    "com.google.android.apps.docs": "Google Docs",
    "com.google.android.apps.messaging": "Google Messages",
    "com.google.android.calendar": "Google Calendar",
    "com.google.android.googlequicksearchbox": "Google",
    "com.google.android.contacts": "Google Contacts",
    "com.google.android.dialer": "Google Phone",
    "com.google.android.apps.labs.language.tailwind": "NotebookLM",
    # System
    "com.android.settings": "Settings",
    "com.android.contacts": "Contacts",
    "com.android.dialer": "Phone",
    "com.android.mms": "Messages",
    "com.android.camera2": "Camera",
    "com.android.gallery3d": "Gallery",
    "com.android.calculator2": "Calculator",
    "com.android.calendar": "Calendar",
    "com.android.deskclock": "Clock",
    "com.android.documentsui": "Files",
    "com.android.vending": "Play Store",
    "com.android.email": "Email",
    # OPPO/ColorOS System
    "com.coloros.soundrecorder": "录音/Sound Recorder",
    "com.coloros.filemanager": "文件管理/File Manager",
    "com.coloros.weather2": "天气/Weather",
    "com.coloros.calendar": "日历/Calendar",
    "com.coloros.calculator": "计算器/Calculator",
    "com.coloros.compass2": "指南针/Compass",
    "com.coloros.alarmclock": "闹钟/Alarm Clock",
    "com.coloros.note": "备忘录/Notes",
    "com.coloros.translate": "翻译/Translate",
    "com.coloros.backuprestore": "备份与恢复/Backup",
    "com.coloros.gallery3d": "相册/Gallery",
    "com.coloros.camera": "相机/Camera",
    "com.coloros.phonemanager": "手机管家/Phone Manager",
    "com.coloros.safecenter": "安全中心/Security Center",
    "com.coloros.oshare": "互传/OShare",
    "com.heytap.browser": "浏览器/Browser",
    "com.heytap.music": "音乐/Music",
    "com.heytap.themestore": "主题商店/Theme Store",
    "com.nearme.gamecenter": "游戏中心/Game Center",
    "com.oppo.market": "应用商店/App Store",
    "com.oppo.quicksearchbox": "搜索/Search",
    # Developer & Tools
    "com.github.android": "GitHub",
    "org.zotero.android": "Zotero",
    "com.server.auditor.ssh.client": "Termius",
    "com.quark.browser": "夸克/Quark",
    "mark.via": "Via Browser",
    # VPN & Security
    "com.tailscale.ipn": "Tailscale",
    "com.github.metacubex.clash.meta": "Clash Meta",
    "com.oray.sunlogin": "向日葵/Sunlogin",
    "com.sangfor.atrust": "aTrust",
    "com.azure.authenticator": "MS Authenticator",
    "com.duosecurity.duomobile": "Duo Mobile",
    # Telecom
    "com.ct.client": "中国电信/China Telecom",
    "com.greenpoint.android.mc10086.activity": "中国移动/China Mobile",
    "com.redteamobile.roaming": "红茶移动/RedTea",
    # Transit
    "com.szt.pay": "深圳通/SZT",
    "com.lingnanpass": "岭南通/Lingnan Pass",
    # Health
    "com.huawei.health": "华为健康/Huawei Health",
    "com.mi.health": "小米健康/Mi Health",
    "com.leoao.fitness": "乐刻运动/Leoao",
    # Gaming
    "com.valvesoftware.android.steam.community": "Steam",
    "com.megacrit.cardcrawl": "Slay the Spire",
    "com.playstack.balatro.android": "Balatro",
    "com.scee.psxandroid": "PlayStation",
    "com.epicgames.portal": "Epic Games",
    "com.max.xiaoheihe": "小黑盒/Xiaoheihe",
    # Other
    "com.wisentsoft.chinapost.android": "中国邮政/China Post",
    "com.fcbox.hiveconsumer": "丰巢/Hive Box",
    "com.cxincx.xxjz": "随手记/Suishouji",
    "com.tplink.ipc": "TP-Link Tapo",
    "io.heckel.ntfy": "ntfy",
    "com.sohu.inputmethod.sogouoem": "搜狗输入法/Sogou",
    "com.podcast.podcasts": "Podcasts",
    "com.jmchn.typhoon": "台风追踪/Typhoon",
    "com.netease.uuremote": "UU加速器/UU Booster",
    "cn.com.chsi.chsiapp": "学信网/CHSI",
    "com.incon.timetable": "课程表/Timetable",
    "cn.edu.hit.welink": "WeLink",
}

# Manual aliases that cannot be derived from display names
_ANDROID_APP_ALIASES_BASE: dict[str, str] = {
    "android settings": "com.android.settings",
    "system settings": "com.android.settings",
    "device settings": "com.android.settings",
    "phone settings": "com.android.settings",
    "com.google.android.settings.intelligence": "com.android.settings",
    "com.android.settings.intelligence": "com.android.settings",
    "google mail": "com.google.android.gm",
    "google gmail": "com.google.android.gm",
    "google chrome": "com.android.chrome",
    "wechat": "com.tencent.mm",
    "alipay": "com.eg.android.AlipayGphone",
    "taobao": "com.taobao.taobao",
    "jd": "com.jingdong.app.mall",
    "jingdong": "com.jingdong.app.mall",
    "meituan": "com.sankuai.meituan",
    "douyin": "com.ss.android.ugc.aweme",
    "tiktok": "com.ss.android.ugc.aweme",
    "bilibili": "tv.danmaku.bili",
    "didi": "com.sdu.didi.psnger",
    "weibo": "com.sina.weibo",
    "zhihu": "com.zhihu.android",
    "redbook": "com.xingin.xhs",
    "rednote": "com.xingin.xhs",
    "xiaohongshu": "com.xingin.xhs",
    "pinduoduo": "com.xunmeng.pinduoduo",
    "xianyu": "com.taobao.idlefish",
    "ctrip": "ctrip.android.view",
    "lark": "com.ss.android.lark",
    "feishu": "com.ss.android.lark",
    "wecom": "com.tencent.wework",
    "weread": "com.tencent.weread",
    "amap": "com.autonavi.minimap",
    "gaode": "com.autonavi.minimap",
    "play store": "com.android.vending",
    "google play": "com.android.vending",
    "twitter": "com.twitter.android",
}


def _build_android_aliases() -> dict[str, str]:
    """Build reverse lookup: display name parts -> package name."""
    aliases: dict[str, str] = {}
    for package, display in _ANDROID_PACKAGE_DISPLAY_NAMES.items():
        # Add each "/" separated part as an alias
        for part in display.split("/"):
            key = part.strip().lower()
            if key and key not in aliases:
                aliases[key] = package
        # Add the full display string
        full = display.strip().lower()
        if full not in aliases:
            aliases[full] = package
    # Manual aliases take priority
    aliases.update(_ANDROID_APP_ALIASES_BASE)
    return aliases


_ANDROID_APP_ALIASES = _build_android_aliases()


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_gui_skill_store_root(workspace: Path) -> Path:
    return Path(workspace) / GUI_SKILLS_DIRNAME


def annotate_android_apps(packages: list[str]) -> list[str]:
    """Annotate package names with human-readable display names.

    Only packages with a known display name are included; unmapped packages are
    silently dropped.  This keeps the system prompt focused on apps the model can
    name and launch, while ``resolve_android_package()`` handles the package name
    lookup at execution time.

    Returns a list like ``["美团/Meituan: com.sankuai.meituan"]``.
    """
    result: list[str] = []
    for pkg in packages:
        display = _ANDROID_PACKAGE_DISPLAY_NAMES.get(pkg)
        if display:
            result.append(f"{display}: {pkg}")
    return result


def resolve_android_package(app_text: str) -> str:
    """Resolve a human-readable app name to its Android package name.

    Returns the matching package name if found, otherwise the input unchanged.
    """
    cleaned = " ".join((app_text or "").strip().strip("\"'").split())
    if not cleaned:
        return app_text or ""
    lowered = cleaned.lower()
    if lowered in _ANDROID_APP_ALIASES:
        return _ANDROID_APP_ALIASES[lowered]
    return cleaned


# ---------------------------------------------------------------------------
# Bundle ID <-> display name mapping for iOS
# ---------------------------------------------------------------------------

_IOS_BUNDLE_DISPLAY_NAMES: dict[str, str] = {
    # Social & Communication
    "com.tencent.xin": "WeChat",
    "com.tencent.mqq": "QQ",
    "com.sina.weibo": "Weibo",
    "com.zhihu.ios": "Zhihu",
    "com.xingin.discover": "RedNote",
    "com.atebits.Tweetie2": "X/Twitter",
    "net.whatsapp.WhatsApp": "WhatsApp",
    "ph.telegra.Telegraph": "Telegram",
    "com.facebook.Facebook": "Facebook",
    "com.bilibili.bilibili": "Bilibili",
    # Shopping & Food
    "com.taobao.taobao4iphone": "Taobao",
    "com.jingdong.app.iphone": "JD",
    "com.xunmeng.pinduoduo": "Pinduoduo",
    "com.taobao.fleamarket": "Xianyu",
    "com.meituan.imeituan": "Meituan",
    "com.dianping.dpscope": "Dianping",
    # Transport
    "com.xiaojukeji.didi": "DiDi",
    "com.autonavi.amap": "Amap",
    # Travel
    "ctrip.com": "Ctrip",
    "com.12306": "12306",
    # Finance
    "com.alipay.iphoneclient": "Alipay",
    # Entertainment
    "com.ss.iphone.ugc.Aweme": "Douyin",
    "com.netease.cloudmusic": "NetEase Music",
    "com.google.ios.youtube": "YouTube",
    # Work & Productivity
    "com.ss.iphone.lark": "Lark",
    "com.tencent.wework": "WeCom",
    "com.tencent.tgmeeting": "VooV",
    # AI
    "com.openai.chat": "ChatGPT",
    "com.deepseek.chat": "DeepSeek",
    # Reading
    "com.tencent.weread": "WeRead",
    # Google
    "com.google.chrome.ios": "Chrome",
    "com.google.Gmail": "Gmail",
    "com.google.Maps": "Google Maps",
    # System
    "com.apple.Preferences": "Settings",
    "com.apple.mobilesafari": "Safari",
    "com.apple.mobilemail": "Mail",
    "com.apple.mobilenotes": "Notes",
    "com.apple.reminders": "Reminders",
    "com.apple.Maps": "Apple Maps",
    "com.apple.camera": "Camera",
    "com.apple.mobileslideshow": "Photos",
    "com.apple.calculator": "Calculator",
    "com.apple.mobiletimer": "Clock",
    "com.apple.weather": "Weather",
    "com.apple.AppStore": "App Store",
    "com.apple.iBooks": "Books",
    "com.apple.Health": "Health",
    "com.apple.Fitness": "Fitness",
    "com.apple.MobileStore": "Apple Store",
    "com.apple.Music": "Music",
    "com.apple.podcasts": "Podcasts",
    "com.apple.tv": "Apple TV",
    "com.apple.DocumentsApp": "Files",
    "com.apple.mobilephone": "Phone",
    "com.apple.MobileSMS": "Messages",
    "com.apple.facetime": "FaceTime",
}

# Manual aliases for iOS that cannot be derived from display names alone
_IOS_APP_ALIASES_BASE: dict[str, str] = {
    "ios settings": "com.apple.Preferences",
    "iphone settings": "com.apple.Preferences",
    "system settings": "com.apple.Preferences",
    "设置": "com.apple.Preferences",
    "系统设置": "com.apple.Preferences",
    "wechat": "com.tencent.xin",
    "weixin": "com.tencent.xin",
    "微信": "com.tencent.xin",
    "qq": "com.tencent.mqq",
    "alipay": "com.alipay.iphoneclient",
    "支付宝": "com.alipay.iphoneclient",
    "taobao": "com.taobao.taobao4iphone",
    "淘宝": "com.taobao.taobao4iphone",
    "jd": "com.jingdong.app.iphone",
    "jingdong": "com.jingdong.app.iphone",
    "京东": "com.jingdong.app.iphone",
    "meituan": "com.meituan.imeituan",
    "美团": "com.meituan.imeituan",
    "大众点评": "com.dianping.dpscope",
    "douyin": "com.ss.iphone.ugc.Aweme",
    "抖音": "com.ss.iphone.ugc.Aweme",
    "tiktok": "com.ss.iphone.ugc.Aweme",
    "bilibili": "com.bilibili.bilibili",
    "哔哩哔哩": "com.bilibili.bilibili",
    "b站": "com.bilibili.bilibili",
    "didi": "com.xiaojukeji.didi",
    "滴滴": "com.xiaojukeji.didi",
    "weibo": "com.sina.weibo",
    "微博": "com.sina.weibo",
    "zhihu": "com.zhihu.ios",
    "知乎": "com.zhihu.ios",
    "redbook": "com.xingin.discover",
    "rednote": "com.xingin.discover",
    "xiaohongshu": "com.xingin.discover",
    "小红书": "com.xingin.discover",
    "pinduoduo": "com.xunmeng.pinduoduo",
    "拼多多": "com.xunmeng.pinduoduo",
    "xianyu": "com.taobao.fleamarket",
    "闲鱼": "com.taobao.fleamarket",
    "ctrip": "ctrip.com",
    "携程": "ctrip.com",
    "lark": "com.ss.iphone.lark",
    "feishu": "com.ss.iphone.lark",
    "飞书": "com.ss.iphone.lark",
    "wecom": "com.tencent.wework",
    "企业微信": "com.tencent.wework",
    "腾讯会议": "com.tencent.tgmeeting",
    "weread": "com.tencent.weread",
    "微信读书": "com.tencent.weread",
    "amap": "com.autonavi.amap",
    "gaode": "com.autonavi.amap",
    "高德": "com.autonavi.amap",
    "高德地图": "com.autonavi.amap",
    "twitter": "com.atebits.Tweetie2",
    "x": "com.atebits.Tweetie2",
    "chatgpt": "com.openai.chat",
    "deepseek": "com.deepseek.chat",
    "youtube": "com.google.ios.youtube",
    "safari": "com.apple.mobilesafari",
    "chrome": "com.google.chrome.ios",
    "gmail": "com.google.Gmail",
    "google maps": "com.google.Maps",
    "相机": "com.apple.camera",
    "照片": "com.apple.mobileslideshow",
    "备忘录": "com.apple.mobilenotes",
    "提醒事项": "com.apple.reminders",
    "地图": "com.apple.Maps",
    "天气": "com.apple.weather",
    "文件": "com.apple.DocumentsApp",
    "电话": "com.apple.mobilephone",
    "短信": "com.apple.MobileSMS",
}


def _build_ios_aliases() -> dict[str, str]:
    """Build reverse lookup: display name parts -> bundle ID."""
    aliases: dict[str, str] = {}
    for bundle_id, display in _IOS_BUNDLE_DISPLAY_NAMES.items():
        # Add each "/" separated part as an alias
        for part in display.split("/"):
            key = part.strip().lower()
            if key and key not in aliases:
                aliases[key] = bundle_id
        # Add the full display string
        full = display.strip().lower()
        if full not in aliases:
            aliases[full] = bundle_id
    # Manual aliases take priority
    aliases.update(_IOS_APP_ALIASES_BASE)
    return aliases


_IOS_APP_ALIASES = _build_ios_aliases()


def annotate_ios_apps(bundle_ids: list[str]) -> list[str]:
    """Annotate iOS bundle IDs with human-readable display names.

    Only bundle IDs with a known display name are included; unmapped entries are
    silently dropped.  This keeps the system prompt focused on apps the model can
    name and launch, while ``resolve_ios_bundle()`` handles the lookup at execution time.

    Returns a list like ``["WeChat: com.tencent.xin"]``.
    """
    result: list[str] = []
    for bundle_id in bundle_ids:
        display = _IOS_BUNDLE_DISPLAY_NAMES.get(bundle_id)
        if display:
            result.append(f"{display}: {bundle_id}")
    return result


def resolve_ios_bundle(app_text: str) -> str:
    """Resolve a human-readable app name to its iOS bundle ID.

    Returns the matching bundle ID if found, otherwise the input unchanged.
    """
    cleaned = " ".join((app_text or "").strip().strip("\"'").split())
    if not cleaned:
        return app_text or ""
    lowered = cleaned.lower()
    if lowered in _IOS_APP_ALIASES:
        return _IOS_APP_ALIASES[lowered]
    return cleaned


def normalize_app_identifier(platform: str, app: str) -> str:
    cleaned = " ".join((app or "").strip().strip("\"'").split())
    if not cleaned:
        return "unknown"

    platform_key = (platform or "").strip().lower()
    lowered = cleaned.lower()

    if platform_key == "android":
        if lowered in _ANDROID_APP_ALIASES:
            return _ANDROID_APP_ALIASES[lowered]
        if "." in cleaned:
            return lowered

    elif platform_key == "ios":
        if lowered in _IOS_APP_ALIASES:
            return _IOS_APP_ALIASES[lowered]
        # Already a bundle ID (contains dots in reverse-domain format)
        if "." in cleaned:
            return cleaned

    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug or "unknown"


def normalize_skill_app(skill: Skill) -> Skill:
    normalized_app = normalize_app_identifier(skill.platform, skill.app)
    if normalized_app == skill.app:
        return skill
    return replace(skill, app=normalized_app)
