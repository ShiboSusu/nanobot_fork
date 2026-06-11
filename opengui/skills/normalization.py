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
    "com.baidu.BaiduMap": "百度地图/Baidu Maps",
    "com.tencent.map": "腾讯地图/Tencent Maps",
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
    "com.ximalaya.ting.android": "喜马拉雅/Ximalaya/Himalaya",
    "com.tencent.qqmusic": "QQ音乐/QQ Music",
    "com.kugou.android": "酷狗音乐/Kugou Music",
    "com.tencent.karaoke": "全民K歌/WeSing",
    "com.qiyi.video": "爱奇艺/iQIYI",
    "com.tencent.qqlive": "腾讯视频/Tencent Video",
    "com.youku.phone": "优酷/Youku",
    "com.smile.gifmaker": "快手/Kuaishou",
    "com.kuaishou.nebula": "快手极速版/Kuaishou Lite",
    "com.ss.android.article.news": "今日头条/Toutiao",
    "com.google.android.youtube": "YouTube",
    "com.bytedance.dreamina": "即梦/Dreamina",
    # Work & Productivity
    "com.ss.android.lark": "飞书/Lark",
    "com.alibaba.android.rimet": "钉钉/DingTalk",
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
    "com.dragon.read": "番茄小说/Fanqie Novel",
    "com.qidian.QDReader": "起点读书/Qidian",
    "com.baidu.searchbox": "百度/Baidu",
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
    "com.tencent.mtt": "QQ浏览器/QQ Browser",
    "com.UCMobile": "UC浏览器/UC Browser",
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
    # MobileWorld emulator apps.  The task text and LLM-generated skills often
    # use generic/AOSP names, while the benchmark image runs branded or forked
    # packages.  Keep these aliases canonical so app-filtered skill reuse does
    # not miss otherwise relevant skills.
    "mail": "com.gmailclone",
    "mail app": "com.gmailclone",
    "email": "com.gmailclone",
    "email app": "com.gmailclone",
    "gmail": "com.gmailclone",
    "gmail app": "com.gmailclone",
    "gmailclone": "com.gmailclone",
    "gmail clone": "com.gmailclone",
    "google mail": "com.gmailclone",
    "google gmail": "com.gmailclone",
    "com.google.android.gm": "com.gmailclone",
    "messages": "com.google.android.apps.messaging",
    "messages app": "com.google.android.apps.messaging",
    "messaging": "com.google.android.apps.messaging",
    "messaging app": "com.google.android.apps.messaging",
    "sms": "com.google.android.apps.messaging",
    "sms app": "com.google.android.apps.messaging",
    "text message": "com.google.android.apps.messaging",
    "text messages": "com.google.android.apps.messaging",
    "com.android.mms": "com.google.android.apps.messaging",
    "com.android.messaging": "com.google.android.apps.messaging",
    "calendar": "org.fossify.calendar",
    "calendar app": "org.fossify.calendar",
    "fossify calendar": "org.fossify.calendar",
    "com.android.calendar": "org.fossify.calendar",
    "com.google.android.calendar": "org.fossify.calendar",
    "files": "com.google.android.documentsui",
    "files app": "com.google.android.documentsui",
    "file manager": "com.google.android.documentsui",
    "documents": "com.google.android.documentsui",
    "documentsui": "com.google.android.documentsui",
    "downloads": "com.google.android.documentsui",
    "downloads folder": "com.google.android.documentsui",
    "download folder": "com.google.android.documentsui",
    "download directory": "com.google.android.documentsui",
    "files-documents": "com.google.android.documentsui",
    "com.android.documentsui": "com.google.android.documentsui",
    "contacts": "com.google.android.contacts",
    "contacts app": "com.google.android.contacts",
    "phone contacts": "com.google.android.contacts",
    "com.android.contacts": "com.google.android.contacts",
    "phone": "com.google.android.dialer",
    "phone app": "com.google.android.dialer",
    "dialer": "com.google.android.dialer",
    "com.android.dialer": "com.google.android.dialer",
    "clock": "com.google.android.deskclock",
    "clock app": "com.google.android.deskclock",
    "deskclock": "com.google.android.deskclock",
    "desk clock": "com.google.android.deskclock",
    "google clock": "com.google.android.deskclock",
    "alarm": "com.google.android.deskclock",
    "alarm clock": "com.google.android.deskclock",
    "com.android.deskclock": "com.google.android.deskclock",
    "mattermost": "com.mattermost.rnbeta",
    "mattermost app": "com.mattermost.rnbeta",
    "mattermost beta": "com.mattermost.rnbeta",
    "mattermost rnbeta": "com.mattermost.rnbeta",
    "com.mattermost": "com.mattermost.rnbeta",
    "com.mattermost.rn": "com.mattermost.rnbeta",
    "slack": "com.mattermost.rnbeta",
    "mastodon": "org.joinmastodon.android.mastodon",
    "mastodon app": "org.joinmastodon.android.mastodon",
    "open talk": "org.joinmastodon.android.mastodon",
    "opentalk": "org.joinmastodon.android.mastodon",
    "org.joinmastodon.android": "org.joinmastodon.android.mastodon",
    "taodian": "com.testmall.app",
    "tao dian": "com.testmall.app",
    "淘店": "com.testmall.app",
    "taobao": "com.testmall.app",
    "com.taobao.taobao": "com.testmall.app",
    "gallery": "gallery.photomanager.picturegalleryapp.imagegallery",
    "gallery app": "gallery.photomanager.picturegalleryapp.imagegallery",
    "photo gallery": "gallery.photomanager.picturegalleryapp.imagegallery",
    "pictures": "gallery.photomanager.picturegalleryapp.imagegallery",
    "com.android.gallery3d": "gallery.photomanager.picturegalleryapp.imagegallery",
    "docreader": "at.tomtasche.reader",
    "doc reader": "at.tomtasche.reader",
    "document reader": "at.tomtasche.reader",
    "open document reader": "at.tomtasche.reader",
    "opendocument reader": "at.tomtasche.reader",
    "pdf reader": "at.tomtasche.reader",
    "reader": "at.tomtasche.reader",
    "reader app": "at.tomtasche.reader",
    "android settings": "com.android.settings",
    "system settings": "com.android.settings",
    "device settings": "com.android.settings",
    "phone settings": "com.android.settings",
    "com.google.android.settings.intelligence": "com.android.settings",
    "com.android.settings.intelligence": "com.android.settings",
    "google chrome": "com.android.chrome",
    "chrome browser": "com.android.chrome",
    "launcher": "com.google.android.apps.nexuslauncher",
    "nexuslauncher": "com.google.android.apps.nexuslauncher",
    "pixel launcher": "com.google.android.apps.nexuslauncher",
    "intentresolver": "com.android.intentresolver",
    "intent resolver": "com.android.intentresolver",
    "wechat": "com.tencent.mm",
    "weixin": "com.tencent.mm",
    "alipay": "com.eg.android.AlipayGphone",
    "zhifubao": "com.eg.android.AlipayGphone",
    "jd": "com.jingdong.app.mall",
    "jingdong": "com.jingdong.app.mall",
    "meituan": "com.sankuai.meituan",
    "douyin": "com.ss.android.ugc.aweme",
    "tiktok": "com.ss.android.ugc.aweme",
    "bilibili": "tv.danmaku.bili",
    "bilibili app": "tv.danmaku.bili",
    "bili": "tv.danmaku.bili",
    "b站": "tv.danmaku.bili",
    "b 站": "tv.danmaku.bili",
    "哔哩哔哩动画": "tv.danmaku.bili",
    "哔站": "tv.danmaku.bili",
    "小破站": "tv.danmaku.bili",
    "youtube app": "com.google.android.youtube",
    "yt": "com.google.android.youtube",
    "油管": "com.google.android.youtube",
    "ximalaya": "com.ximalaya.ting.android",
    "himalaya": "com.ximalaya.ting.android",
    "xmly": "com.ximalaya.ting.android",
    "喜马拉雅fm": "com.ximalaya.ting.android",
    "qqmusic": "com.tencent.qqmusic",
    "kugou": "com.kugou.android",
    "iqiyi": "com.qiyi.video",
    "kuaishou": "com.smile.gifmaker",
    "kwai": "com.smile.gifmaker",
    "toutiao": "com.ss.android.article.news",
    "didi": "com.sdu.didi.psnger",
    "weibo": "com.sina.weibo",
    "zhihu": "com.zhihu.android",
    "xhs": "com.xingin.xhs",
    "redbook": "com.xingin.xhs",
    "rednote": "com.xingin.xhs",
    "xiaohongshu": "com.xingin.xhs",
    "pinduoduo": "com.xunmeng.pinduoduo",
    "xianyu": "com.taobao.idlefish",
    "ctrip": "ctrip.android.view",
    "ctrip.com": "ctrip.android.view",
    "携程旅行": "ctrip.android.view",
    "lark": "com.ss.android.lark",
    "feishu": "com.ss.android.lark",
    "dingtalk": "com.alibaba.android.rimet",
    "wecom": "com.tencent.wework",
    "voov": "com.tencent.wemeet.app",
    "tencent meeting": "com.tencent.wemeet.app",
    "tencent docs": "com.tencent.docs",
    "weread": "com.tencent.weread",
    "fanqie": "com.dragon.read",
    "qidian": "com.qidian.QDReader",
    "baidu": "com.baidu.searchbox",
    "maps": "com.google.android.apps.maps",
    "maps app": "com.google.android.apps.maps",
    "map": "com.google.android.apps.maps",
    "amap": "com.autonavi.minimap",
    "gaode": "com.autonavi.minimap",
    "google map": "com.google.android.apps.maps",
    "google maps": "com.google.android.apps.maps",
    "baidu map": "com.baidu.BaiduMap",
    "baidu maps": "com.baidu.BaiduMap",
    "tencent map": "com.tencent.map",
    "tencent maps": "com.tencent.map",
    "play store": "com.android.vending",
    "google play": "com.android.vending",
    "twitter": "com.twitter.android",
}


_ANDROID_ADB_APP_ALIASES_BASE: dict[str, str] = {
    # Real-device package names. Keep these separate from MobileWorld aliases:
    # benchmark packages may be synthetic, while ADB commands such as
    # ``pm path`` and ``monkey -p`` must use packages installed on the device.
    "mastodon": "org.joinmastodon.android",
    "mastodon app": "org.joinmastodon.android",
    "org.joinmastodon.android": "org.joinmastodon.android",
    "org.joinmastodon.android.mastodon": "org.joinmastodon.android",
}


_ANDROID_ALIAS_SUFFIXES = (
    " app",
    " application",
    " 应用",
    " 软件",
    " 客户端",
    "app",
    "应用",
    "软件",
    "客户端",
)
_ANDROID_ALIAS_PREFIXES = (
    "打开",
    "启动",
    "开启",
    "运行",
    "进入",
    "open ",
    "launch ",
    "start ",
)


def _android_display_aliases() -> dict[str, str]:
    """Build reverse lookup from display name parts to package names."""
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
    return aliases


def _build_android_aliases() -> dict[str, str]:
    aliases = _android_display_aliases()
    # Manual/synthetic aliases take priority for skill storage and MobileWorld.
    aliases.update(_ANDROID_APP_ALIASES_BASE)
    return aliases


def _build_android_adb_aliases() -> dict[str, str]:
    aliases = _android_display_aliases()
    # Real-device aliases take priority for ADB runtime/package inspection.
    aliases.update(_ANDROID_ADB_APP_ALIASES_BASE)
    return aliases


_ANDROID_APP_ALIASES = _build_android_aliases()
_ANDROID_ADB_APP_ALIASES = _build_android_adb_aliases()
_ANDROID_TEXT_ALIAS_DENYLIST = {"search", "搜索", "video", "视频"}


def _lookup_android_alias_from(lowered: str, aliases: dict[str, str]) -> str | None:
    candidates = [lowered]
    for prefix in _ANDROID_ALIAS_PREFIXES:
        if lowered.startswith(prefix) and len(lowered) > len(prefix):
            candidates.append(lowered[len(prefix):].strip())
    for candidate in candidates:
        package = aliases.get(candidate)
        if package:
            return package
        for suffix in _ANDROID_ALIAS_SUFFIXES:
            if candidate.endswith(suffix) and len(candidate) > len(suffix):
                stem = candidate[: -len(suffix)].strip()
                if stem:
                    package = aliases.get(stem)
                    if package:
                        return package
    return None


def _lookup_android_alias(lowered: str) -> str | None:
    return _lookup_android_alias_from(lowered, _ANDROID_APP_ALIASES)


def _lookup_android_adb_alias(lowered: str) -> str | None:
    return _lookup_android_alias_from(lowered, _ANDROID_ADB_APP_ALIASES)


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
    package = _lookup_android_alias(lowered)
    if package:
        return package
    return cleaned


def find_android_apps_in_text(text: str, *, max_apps: int = 5) -> list[str]:
    """Return Android app packages mentioned inside free-form text."""
    lowered = " ".join((text or "").strip().lower().split())
    if not lowered:
        return []
    compact = re.sub(r"\s+", "", lowered)
    matches: list[tuple[int, str, str]] = []
    for alias, package in _ANDROID_APP_ALIASES.items():
        alias_lower = alias.strip().lower()
        if alias_lower in _ANDROID_TEXT_ALIAS_DENYLIST:
            continue
        if len(alias_lower) < 2:
            continue
        if _android_alias_occurs(lowered, compact, alias_lower):
            matches.append((len(alias_lower), alias_lower, package))
    if not matches:
        return []
    matches.sort(reverse=True)
    packages: list[str] = []
    for _, _, package in matches:
        if package not in packages:
            packages.append(package)
        if len(packages) >= max_apps:
            break
    return packages


def find_android_app_in_text(text: str) -> str | None:
    """Return the best Android app package mentioned inside free-form text."""
    packages = find_android_apps_in_text(text, max_apps=1)
    return packages[0] if packages else None


def _android_alias_occurs(text: str, compact_text: str, alias: str) -> bool:
    if "." in alias:
        return alias in text
    compact_alias = re.sub(r"\s+", "", alias)
    if re.search(r"[\u4e00-\u9fff]", compact_alias):
        return compact_alias in compact_text
    return re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text) is not None


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
    "tv.danmaku.bilianime": "Bilibili/哔哩哔哩",
    "com.bilibili.bilibili": "Bilibili/哔哩哔哩",
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
    "alipay": "com.alipay.iphoneclient",
    "支付宝": "com.alipay.iphoneclient",
    "taobao": "com.taobao.taobao4iphone",
    "淘宝": "com.taobao.taobao4iphone",
    "jd": "com.jingdong.app.iphone",
    "jingdong": "com.jingdong.app.iphone",
    "京东": "com.jingdong.app.iphone",
    "meituan": "com.meituan.imeituan",
    "美团": "com.meituan.imeituan",
    "douyin": "com.ss.iphone.ugc.Aweme",
    "抖音": "com.ss.iphone.ugc.Aweme",
    "tiktok": "com.ss.iphone.ugc.Aweme",
    "bilibili": "tv.danmaku.bilianime",
    "哔哩哔哩": "tv.danmaku.bilianime",
    "b站": "tv.danmaku.bilianime",
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
    "ctrip": "ctrip.com",
    "携程": "ctrip.com",
    "ctrip.android.view": "ctrip.com",
    "携程旅行": "ctrip.com",
    "12306": "com.12306",
    "铁路12306": "com.12306",
    "lark": "com.ss.iphone.lark",
    "feishu": "com.ss.iphone.lark",
    "wecom": "com.tencent.wework",
    "weread": "com.tencent.weread",
    "amap": "com.autonavi.amap",
    "gaode": "com.autonavi.amap",
    "高德": "com.autonavi.amap",
    "高德地图": "com.autonavi.amap",
    "twitter": "com.atebits.Tweetie2",
    "x": "com.atebits.Tweetie2",
    "chatgpt": "com.openai.chat",
    "deepseek": "com.deepseek.chat",
    "youtube": "com.google.ios.youtube",
    "网易云音乐": "com.netease.cloudmusic",
    "云音乐": "com.netease.cloudmusic",
    "apple store": "com.apple.MobileStore",
    "apple music": "com.apple.Music",
    "apple podcasts": "com.apple.podcasts",
    "apple books": "com.apple.iBooks",
    "apple tv": "com.apple.tv",
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


def _clean_ios_app_text(app_text: str) -> str:
    return " ".join((app_text or "").strip().strip("\"'“”‘’").split())


def _canonical_ios_app_key(app_text: str) -> str:
    cleaned = _clean_ios_app_text(app_text).casefold()
    return re.sub(r"[\s._\-/:：·'\"“”‘’()（）]+", "", cleaned)


def _ios_lookup_keys(app_text: str) -> tuple[str, ...]:
    cleaned = _clean_ios_app_text(app_text)
    lowered = cleaned.casefold()
    canonical = _canonical_ios_app_key(cleaned)
    keys: list[str] = []
    for key in (lowered, canonical):
        if key and key not in keys:
            keys.append(key)
    return tuple(keys)


def _add_ios_alias(
    aliases: dict[str, str],
    alias: str,
    bundle_id: str,
    *,
    overwrite: bool = False,
) -> None:
    for key in _ios_lookup_keys(alias):
        if overwrite:
            aliases[key] = bundle_id
        else:
            aliases.setdefault(key, bundle_id)


def _looks_like_ios_bundle(value: str) -> bool:
    cleaned = _clean_ios_app_text(value)
    return "." in cleaned and " " not in cleaned and "\t" not in cleaned


def _parse_ios_app_entry(entry: str) -> tuple[str | None, str | None]:
    cleaned = _clean_ios_app_text(entry)
    if not cleaned:
        return None, None
    for separator in (": ", "：", ":"):
        if separator in cleaned:
            display, bundle_id = cleaned.rsplit(separator, 1)
            bundle_id = bundle_id.strip()
            if display.strip() and _looks_like_ios_bundle(bundle_id):
                return display.strip(), bundle_id
    if _looks_like_ios_bundle(cleaned):
        return None, cleaned
    return cleaned, None


def _ios_display_aliases(display: str) -> tuple[str, ...]:
    aliases = [_clean_ios_app_text(display)]
    aliases.extend(part.strip() for part in re.split(r"[/／|｜]", display) if part.strip())
    aliases.extend(
        suffix.strip()
        for suffix in (display.removesuffix(" App"), display.removesuffix(" app"), display.removesuffix("应用"))
        if suffix.strip()
    )
    result: list[str] = []
    for alias in aliases:
        if alias and alias not in result:
            result.append(alias)
    return tuple(result)


def _ios_related_static_aliases(display_aliases: tuple[str, ...]) -> tuple[str, ...]:
    related: list[str] = []
    static_bundles: set[str] = set()
    for alias in display_aliases:
        for key in _ios_lookup_keys(alias):
            static_bundle = _IOS_APP_ALIASES.get(key)
            if static_bundle:
                static_bundles.add(static_bundle)

    for static_bundle in static_bundles:
        display = _IOS_BUNDLE_DISPLAY_NAMES.get(static_bundle)
        if display:
            related.extend(_ios_display_aliases(display))
        related.extend(
            alias
            for alias, bundle_id in _IOS_APP_ALIASES_BASE.items()
            if bundle_id == static_bundle
        )

    deduped: list[str] = []
    for alias in related:
        if alias and alias not in deduped:
            deduped.append(alias)
    return tuple(deduped)


def _build_ios_installed_aliases(installed_apps: list[str] | None) -> tuple[dict[str, str], set[str]]:
    aliases: dict[str, str] = {}
    bundle_ids: set[str] = set()
    for entry in installed_apps or []:
        display, bundle_id = _parse_ios_app_entry(entry)
        if not bundle_id:
            continue
        bundle_ids.add(bundle_id)

        display = display or _IOS_BUNDLE_DISPLAY_NAMES.get(bundle_id)
        if display:
            display_aliases = _ios_display_aliases(display)
            for alias in display_aliases:
                _add_ios_alias(aliases, alias, bundle_id)
            for alias in _ios_related_static_aliases(display_aliases):
                _add_ios_alias(aliases, alias, bundle_id)

        tail = bundle_id.rsplit(".", 1)[-1]
        if len(tail) >= 3:
            _add_ios_alias(aliases, tail, bundle_id)
    return aliases, bundle_ids


def _build_ios_aliases() -> dict[str, str]:
    """Build reverse lookup: display name parts -> bundle ID."""
    aliases: dict[str, str] = {}
    for bundle_id, display in _IOS_BUNDLE_DISPLAY_NAMES.items():
        for part in _ios_display_aliases(display):
            _add_ios_alias(aliases, part, bundle_id)
        _add_ios_alias(aliases, display, bundle_id)
    for alias, bundle_id in _IOS_APP_ALIASES_BASE.items():
        _add_ios_alias(aliases, alias, bundle_id, overwrite=True)
    return aliases


_IOS_APP_ALIASES = _build_ios_aliases()


IOS_BROWSER_BUNDLE_IDS: frozenset[str] = frozenset(
    {
        "com.apple.mobilesafari",
        "com.google.chrome.ios",
    }
)


def is_browser_bundle(bundle_id: str | None) -> bool:
    return bool(bundle_id and bundle_id in IOS_BROWSER_BUNDLE_IDS)


def task_explicitly_allows_browser(text: str | None) -> bool:
    if not text:
        return False
    return any(
        marker in text
        for marker in (
            "浏览器",
            "网页",
            "网页版",
            "Safari",
            "safari",
            "Chrome",
            "chrome",
        )
    )


def annotate_ios_apps(bundle_ids: list[str]) -> list[str]:
    """Annotate iOS bundle IDs with human-readable display names.

    Only bundle IDs with a known display name are included; unmapped entries are
    silently dropped.  This keeps the system prompt focused on apps the model can
    name and launch, while ``resolve_ios_bundle()`` handles the lookup at execution time.

    Returns a list like ``["WeChat: com.tencent.xin"]``.
    """
    result: list[str] = []
    seen: set[str] = set()
    for entry in bundle_ids:
        display, bundle_id = _parse_ios_app_entry(entry)
        if not bundle_id or bundle_id in seen:
            continue
        display = display or _IOS_BUNDLE_DISPLAY_NAMES.get(bundle_id)
        if display:
            result.append(f"{display}: {bundle_id}")
            seen.add(bundle_id)
    return result


def resolve_ios_bundle(app_text: str, installed_apps: list[str] | None = None) -> str:
    """Resolve a human-readable app name to its iOS bundle ID.

    Device-discovered app names take precedence over the static alias table so
    stale aliases do not break launch on devices that use different bundle IDs.
    Returns the matching bundle ID if found, otherwise the input unchanged.
    """
    cleaned = _clean_ios_app_text(app_text)
    if not cleaned:
        return app_text or ""
    installed_aliases, installed_bundle_ids = _build_ios_installed_aliases(installed_apps)
    for key in _ios_lookup_keys(cleaned):
        if key in installed_aliases:
            return installed_aliases[key]
    if _looks_like_ios_bundle(cleaned) and (
        not installed_bundle_ids or cleaned in installed_bundle_ids
    ):
        return cleaned
    candidate_keys = _ios_lookup_keys(cleaned)
    for key in candidate_keys:
        if key in _IOS_APP_ALIASES:
            return _IOS_APP_ALIASES[key]
    canonical_text = _canonical_ios_app_key(cleaned)
    for alias_key, bundle_id in _IOS_APP_ALIASES.items():
        if alias_key and alias_key in canonical_text:
            return bundle_id
    return cleaned


def normalize_app_identifier(platform: str, app: str) -> str:
    cleaned = " ".join((app or "").strip().strip("\"'").split())
    if not cleaned:
        return "unknown"

    platform_key = (platform or "").strip().lower()
    lowered = cleaned.lower()

    if platform_key == "android":
        package = _lookup_android_alias(lowered)
        if package:
            return package
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


def normalize_adb_app_identifier(app: str) -> str:
    """Normalize an Android app identifier for real ADB device commands.

    This intentionally does not use MobileWorld/synthetic aliases. If the input
    is already a package name and has no real-device alias override, it is
    preserved so foreground packages from ``dumpsys`` remain executable.
    """
    cleaned = " ".join((app or "").strip().strip("\"'").split())
    if not cleaned:
        return "unknown"

    lowered = cleaned.lower()
    package = _lookup_android_adb_alias(lowered)
    if package:
        return package
    if "." in cleaned:
        return lowered

    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug or "unknown"


def normalize_skill_app(skill: Skill) -> Skill:
    normalized_app = normalize_app_identifier(skill.platform, skill.app)
    if normalized_app == skill.app:
        return skill
    return replace(skill, app=normalized_app)


def normalize_app_filter(platform: str | None, app: str | None) -> str | None:
    """Normalize *app* for use as a filter key, or return None."""
    if app is None:
        return None
    return normalize_app_identifier(platform or "unknown", app)
