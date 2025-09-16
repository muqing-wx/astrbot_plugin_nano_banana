import asyncio
import base64
import functools
import io
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import aiohttp
from PIL import Image as PILImage

from astrbot import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import At, Image, Reply, Plain
from astrbot.core.platform.astr_message_event import AstrMessageEvent


@register(
    "astrbot_plugin_nano_banana",
    "沐沐沐倾",
    "基于柏拉图api集多种预设风格、自定义图/文生图、智能对话绘画及后台管理于一体的强大AI生图插件。",
    "1.0.1", # 增加卸载清理和LLM工具开关
)
class BananaPlugin(Star):
    class ImageWorkflow:
        def __init__(self, proxy_url: str | None = None):
            if proxy_url:
                logger.info(f"ImageWorkflow 使用代理: {proxy_url}")
            self.session = aiohttp.ClientSession()
            self.proxy = proxy_url

        async def _download_image(self, url: str) -> bytes | None:
            try:
                async with self.session.get(url, proxy=self.proxy, timeout=30) as resp:
                    resp.raise_for_status()
                    return await resp.read()
            except Exception as e:
                logger.error(f"图片下载失败: {e}")
                return None

        async def _get_avatar(self, user_id: str) -> bytes | None:
            if not user_id.isdigit():
                logger.warning(f"无法获取非 QQ 平台或无效 QQ 号 {user_id} 的头像。")
                return None
            avatar_url = f"https://q1.qlogo.cn/g?b=qq&nk={user_id}&s=640"
            return await self._download_image(avatar_url)

        def _extract_first_frame_sync(self, raw: bytes) -> bytes:
            img_io = io.BytesIO(raw)
            try:
                with PILImage.open(img_io) as img:
                    if getattr(img, "is_animated", False):
                        logger.info("检测到动图, 将抽取第一帧进行生成")
                        img.seek(0)
                        first_frame = img.convert("RGBA")
                        out_io = io.BytesIO()
                        first_frame.save(out_io, format="PNG")
                        return out_io.getvalue()
            except Exception as e:
                logger.warning(f"抽取图片帧时发生错误, 将返回原始数据: {e}", exc_info=True)
            return raw

        async def _load_bytes(self, src: str) -> bytes | None:
            raw: bytes | None = None
            loop = asyncio.get_running_loop()
            if Path(src).is_file():
                raw = await loop.run_in_executor(None, Path(src).read_bytes)
            elif src.startswith("http"):
                raw = await self._download_image(src)
            elif src.startswith("base64://"):
                raw = await loop.run_in_executor(None, base64.b64decode, src[9:])
            if not raw:
                return None
            return await loop.run_in_executor(None, self._extract_first_frame_sync, raw)

        async def _get_images_from_segments(self, event: AstrMessageEvent) -> List[bytes]:
            """仅从消息段中提取显式图片（发送、回复、引用）。"""
            images = []
            processed_urls = set()

            async def process_image(seg: Image):
                url_or_file = seg.url or seg.file
                if url_or_file and url_or_file not in processed_urls:
                    if img_bytes := await self._load_bytes(url_or_file):
                        images.append(img_bytes)
                        processed_urls.add(url_or_file)

            for seg in event.message_obj.message:
                if isinstance(seg, Reply) and seg.chain:
                    for s_chain in seg.chain:
                        if isinstance(s_chain, Image):
                            await process_image(s_chain)
            
            for seg in event.message_obj.message:
                if isinstance(seg, Image):
                    await process_image(seg)
            return images

        async def get_explicit_images_from_message(self, event: AstrMessageEvent) -> List[bytes]:
            """供LLM工具使用：只获取消息中明确存在的图片。"""
            images = await self._get_images_from_segments(event)
            if images:
                logger.info(f"LLM工具调用：在此次请求中找到了 {len(images)} 张显式图片。")
            return images

        async def get_all_images(self, event: AstrMessageEvent) -> List[bytes]:
            """供指令使用：获取所有图片，包括作为后备的头像。"""
            images = await self._get_images_from_segments(event)
            if images:
                logger.info(f"指令调用：在此次请求中找到了 {len(images)} 张显式图片。")
                return images

            at_user_id = next((str(s.qq) for s in event.message_obj.message if isinstance(s, At)), None)
            
            if at_user_id:
                if avatar := await self._get_avatar(at_user_id):
                    logger.info(f"指令调用：未找到图片，使用被@用户 {at_user_id} 的头像。")
                    return [avatar]

            if avatar := await self._get_avatar(event.get_sender_id()):
                logger.info(f"指令调用：未找到图片，使用发送者 {event.get_sender_id()} 的头像。")
                return [avatar]

            return []

        async def terminate(self):
            if self.session and not self.session.closed:
                await self.session.close()

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        self.plugin_data_dir = StarTools.get_data_dir()
        self.user_counts_file = self.plugin_data_dir / "user_counts.json"
        self.user_counts: Dict[str, int] = {}
        self.group_counts_file = self.plugin_data_dir / "group_counts.json"
        self.group_counts: Dict[str, int] = {}
        self.key_index = 0
        self.key_lock = asyncio.Lock()
        self.iwf: Optional[BananaPlugin.ImageWorkflow] = None
        self.default_prompts: Dict[str, str] = {}

    async def initialize(self):
        prompts_file = Path(__file__).parent / "prompts.json"
        if prompts_file.exists():
            try:
                content = prompts_file.read_text("utf-8")
                self.default_prompts = json.loads(content)
                logger.info("默认 prompts.json 文件已加载")
            except Exception as e:
                logger.error(f"加载默认 prompts.json 文件失败: {e}", exc_info=True)
        use_proxy = self.conf.get("use_proxy", False)
        proxy_url = self.conf.get("proxy_url") if use_proxy else None
        self.iwf = self.ImageWorkflow(proxy_url)
        await self._load_user_counts()
        await self._load_group_counts()
        logger.info("Nano Banana 生图插件已加载")
        if not self.conf.get("api_keys"):
            logger.warning("NanoBananaPlugin: 未配置任何[生图] API 密钥，插件可能无法工作")

    async def uninstall(self):
        """插件卸载时调用的方法，用于清理资源。"""
        logger.info("正在卸载 Nano Banana 插件，开始清理数据文件...")
        try:
            if self.user_counts_file.exists():
                self.user_counts_file.unlink()
                logger.info(f"已删除用户次数文件: {self.user_counts_file}")
            if self.group_counts_file.exists():
                self.group_counts_file.unlink()
                logger.info(f"已删除群组次数文件: {self.group_counts_file}")
            logger.info("数据文件清理完成。")
        except Exception as e:
            logger.error(f"卸载插件时清理文件失败: {e}", exc_info=True)

    def is_global_admin(self, event: AstrMessageEvent) -> bool:
        admin_ids = self.context.get_config().get("admins_id", [])
        return event.get_sender_id() in admin_ids

    async def _load_user_counts(self):
        if not self.user_counts_file.exists():
            self.user_counts = {}
            return
        loop = asyncio.get_running_loop()
        try:
            content = await loop.run_in_executor(None, self.user_counts_file.read_text, "utf-8")
            data = await loop.run_in_executor(None, json.loads, content)
            if isinstance(data, dict):
                self.user_counts = {str(k): v for k, v in data.items()}
        except Exception as e:
            logger.error(f"加载用户次数文件时发生错误: {e}", exc_info=True)
            self.user_counts = {}

    async def _save_user_counts(self):
        loop = asyncio.get_running_loop()
        try:
            json_data = await loop.run_in_executor(None, functools.partial(json.dumps, self.user_counts, ensure_ascii=False, indent=4))
            await loop.run_in_executor(None, self.user_counts_file.write_text, json_data, "utf-8")
        except Exception as e:
            logger.error(f"保存用户次数文件时发生错误: {e}", exc_info=True)

    def _get_user_count(self, user_id: str) -> int:
        return self.user_counts.get(str(user_id), 0)

    async def _decrease_user_count(self, user_id: str):
        user_id_str = str(user_id)
        count = self._get_user_count(user_id_str)
        if count > 0:
            self.user_counts[user_id_str] = count - 1
            await self._save_user_counts()

    async def _load_group_counts(self):
        if not self.group_counts_file.exists():
            self.group_counts = {}
            return
        loop = asyncio.get_running_loop()
        try:
            content = await loop.run_in_executor(None, self.group_counts_file.read_text, "utf-8")
            data = await loop.run_in_executor(None, json.loads, content)
            if isinstance(data, dict):
                self.group_counts = {str(k): v for k, v in data.items()}
        except Exception as e:
            logger.error(f"加载群组次数文件时发生错误: {e}", exc_info=True)
            self.group_counts = {}

    async def _save_group_counts(self):
        loop = asyncio.get_running_loop()
        try:
            json_data = await loop.run_in_executor(None, functools.partial(json.dumps, self.group_counts, ensure_ascii=False, indent=4))
            await loop.run_in_executor(None, self.group_counts_file.write_text, json_data, "utf-8")
        except Exception as e:
            logger.error(f"保存群组次数文件时发生错误: {e}", exc_info=True)

    def _get_group_count(self, group_id: str) -> int:
        return self.group_counts.get(str(group_id), 0)

    async def _decrease_group_count(self, group_id: str):
        group_id_str = str(group_id)
        count = self._get_group_count(group_id_str)
        if count > 0:
            self.group_counts[group_id_str] = count - 1
            await self._save_group_counts()

    # ------------------- LLM 工具定义 -------------------

    @filter.llm_tool(name="nano_banana_text_to_image")
    async def text_to_image_tool(self, event: AstrMessageEvent, prompt: str):
        """
        文生图工具：当用户意图是“从零开始、仅凭文字描述”来创造一张新图片时使用。
        触发条件：用户的消息中不包含任何显式图片（发送、回复、引用），但有清晰的绘图或创作指令。
        
        使用示例:
        - "画一只猫"
        - "生成一张未来城市的科幻图片"
        - "一个宇航员在月球上骑着马，超现实主义风格"
        - "draw a dog playing a guitar"

        Args:
            prompt (str): 用户的原始文本。必须直接使用，不得进行任何修改、翻译或扩写。
        """
        if not self.conf.get("enable_llm_tools", False):
            logger.debug("LLM工具调用功能已在配置中禁用，跳过 nano_banana_text_to_image 执行。")
            return
        
        logger.info(f"核心LLM触发工具: nano_banana_text_to_image, prompt: {prompt}")
        async for result in self._process_generation_request(event, "自然语言-文生图", require_image=False, natural_prompt=prompt):
            yield result

    @filter.llm_tool(name="nano_banana_image_to_image")
    async def image_to_image_tool(self, event: AstrMessageEvent, prompt: str):
        """
        图生图工具：当用户意图是“基于已有图片”进行修改、变换风格、重绘或二次创作时使用。
        触发条件：用户的消息中必须同时包含显式图片（发送、回复、引用）和描述性的文本指令。

        使用示例:
        - (用户发送一张猫的图片并说): "给它戴上帽子"
        - (用户回复一张风景照并说): "把这张图变成动漫风格"
        - (用户发送一张人物照片并说): "把背景换成星空"
        - (用户引用一张草图并说): "帮我把它细化上色"

        Args:
            prompt (str): 用户的原始修改或创作指令文本。必须直接使用，不得进行任何修改、翻译或扩写。
        """
        if not self.conf.get("enable_llm_tools", False):
            logger.debug("LLM工具调用功能已在配置中禁用，跳过 nano_banana_image_to_image 执行。")
            return

        logger.info(f"核心LLM触发工具: nano_banana_image_to_image, prompt: {prompt}")
        if not self.iwf:
            yield event.plain_result("插件内部错误：ImageWorkflow未初始化。")
            return

        # LLM工具严格使用显式图片
        explicit_images = await self.iwf.get_explicit_images_from_message(event)
        if not explicit_images:
            yield event.plain_result("图生图需要一张图片，但我没有在您的消息中找到。")
            return

        async for result in self._process_generation_request(event, "自然语言-图生图", require_image=True, natural_prompt=prompt, pre_fetched_images=explicit_images):
            yield result

    # ------------------- 命令处理 -------------------

    @filter.command("生图增加用户次数", prefix_optional=True)
    async def on_add_user_counts(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            return
        cmd_text = event.message_str.strip()
        at_seg = next((s for s in event.message_obj.message if isinstance(s, At)), None)
        target_qq, count = None, 0
        if at_seg:
            target_qq = str(at_seg.qq)
            match = re.search(r"(\d+)\s*$", cmd_text)
            if match:
                count = int(match.group(1))
        else:
            match = re.search(r"(\d+)\s+(\d+)", cmd_text)
            if match:
                target_qq, count = match.group(1), int(match.group(2))
        if not target_qq or count <= 0:
            yield event.plain_result('格式错误:\n#生图增加用户次数 @用户 <次数>\n或 #生图增加用户次数 <QQ号> <次数>')
            return
        current_count = self._get_user_count(target_qq)
        self.user_counts[str(target_qq)] = current_count + count
        await self._save_user_counts()
        yield event.plain_result(f"✅ 已为用户 {target_qq} 增加 {count} 次，TA当前剩余 {current_count + count} 次。")

    @filter.command("生图增加群组次数", prefix_optional=True)
    async def on_add_group_counts(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            return
        cmd_text = event.message_str.strip()
        match = re.search(r"(\d+)\s+(\d+)", cmd_text)
        if not match:
            yield event.plain_result('格式错误: #生图增加群组次数 <群号> <次数>')
            return
        target_group, count = match.group(1), int(match.group(2))
        current_count = self._get_group_count(target_group)
        self.group_counts[str(target_group)] = current_count + count
        await self._save_group_counts()
        yield event.plain_result(f"✅ 已为群组 {target_group} 增加 {count} 次，该群当前剩余 {current_count + count} 次。")

    @filter.command("生图查询次数", prefix_optional=True)
    async def on_query_counts(self, event: AstrMessageEvent):
        user_id_to_query = event.get_sender_id()
        if self.is_global_admin(event):
            at_seg = next((s for s in event.message_obj.message if isinstance(s, At)), None)
            if at_seg:
                user_id_to_query = str(at_seg.qq)
            else:
                match = re.search(r"(\d+)", event.message_str)
                if match:
                    user_id_to_query = match.group(1)

        user_count = self._get_user_count(user_id_to_query)
        reply_msg = f"用户 {user_id_to_query} 个人剩余次数为: {user_count}" if user_id_to_query != event.get_sender_id() else f"您好，您当前个人剩余次数为: {user_count}"

        group_id = event.get_group_id()
        if group_id:
            group_count = self._get_group_count(group_id)
            reply_msg += f"\n本群共享剩余次数为: {group_count}"
        yield event.plain_result(reply_msg)

    @filter.command("生图添加key", prefix_optional=True)
    async def on_add_key(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            return
        new_keys = event.message_str.strip().split()
        if not new_keys:
            yield event.plain_result("格式错误，请提供要添加的Key。")
            return
        api_keys = self.conf.get("api_keys", [])
        added_keys = [key for key in new_keys if key not in api_keys]
        api_keys.extend(added_keys)
        await self.conf.set("api_keys", api_keys)
        yield event.plain_result(f"✅ 操作完成，新增 {len(added_keys)} 个Key，当前共 {len(api_keys)} 个。")

    @filter.command("生图key列表", prefix_optional=True)
    async def on_list_keys(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            return
        api_keys = self.conf.get("api_keys", [])
        if not api_keys:
            yield event.plain_result("📝 暂未配置任何 API Key。")
            return
        key_list_str = "\n".join(f"{i + 1}. {key[:8]}...{key[-4:]}" for i, key in enumerate(api_keys))
        yield event.plain_result(f"🔑 API Key 列表:\n{key_list_str}")

    @filter.command("生图删除key", prefix_optional=True)
    async def on_delete_key(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            return
        param = event.message_str.strip()
        api_keys = self.conf.get("api_keys", [])
        if param.lower() == "all":
            count = len(api_keys)
            await self.conf.set("api_keys", [])
            yield event.plain_result(f"✅ 已删除全部 {count} 个 Key。")
        elif param.isdigit() and 1 <= int(param) <= len(api_keys):
            idx = int(param) - 1
            removed_key = api_keys.pop(idx)
            await self.conf.set("api_keys", api_keys)
            yield event.plain_result(f"✅ 已删除 Key: {removed_key[:8]}...")
        else:
            yield event.plain_result("格式错误，请使用 #生图删除key <序号|all>")

    @filter.command("手办化", prefix_optional=True)
    async def on_cmd_figurine(self, event: AstrMessageEvent):
        async for result in self._process_generation_request(event, "手办化", require_image=True): yield result
    @filter.command("手办化2", prefix_optional=True)
    async def on_cmd_figurine2(self, event: AstrMessageEvent):
        async for result in self._process_generation_request(event, "手办化2", require_image=True): yield result
    @filter.command("手办化3", prefix_optional=True)
    async def on_cmd_figurine3(self, event: AstrMessageEvent):
        async for result in self._process_generation_request(event, "手办化3", require_image=True): yield result
    @filter.command("手办化4", prefix_optional=True)
    async def on_cmd_figurine4(self, event: AstrMessageEvent):
        async for result in self._process_generation_request(event, "手办化4", require_image=True): yield result
    @filter.command("手办化5", prefix_optional=True)
    async def on_cmd_figurine5(self, event: AstrMessageEvent):
        async for result in self._process_generation_request(event, "手办化5", require_image=True): yield result
    @filter.command("手办化6", prefix_optional=True)
    async def on_cmd_figurine6(self, event: AstrMessageEvent):
        async for result in self._process_generation_request(event, "手办化6", require_image=True): yield result
    @filter.command("Q版化", prefix_optional=True)
    async def on_cmd_qversion(self, event: AstrMessageEvent):
        async for result in self._process_generation_request(event, "Q版化", require_image=True): yield result
    @filter.command("痛屋化", prefix_optional=True)
    async def on_cmd_painroom(self, event: AstrMessageEvent):
        async for result in self._process_generation_request(event, "痛屋化", require_image=True): yield result
    @filter.command("痛屋化2", prefix_optional=True)
    async def on_cmd_painroom2(self, event: AstrMessageEvent):
        async for result in self._process_generation_request(event, "痛屋化2", require_image=True): yield result
    @filter.command("痛车化", prefix_optional=True)
    async def on_cmd_paincar(self, event: AstrMessageEvent):
        async for result in self._process_generation_request(event, "痛车化", require_image=True): yield result
    @filter.command("cos化", prefix_optional=True)
    async def on_cmd_cos(self, event: AstrMessageEvent):
        async for result in self._process_generation_request(event, "cos化", require_image=True): yield result
    @filter.command("cos自拍", prefix_optional=True)
    async def on_cmd_cos_selfie(self, event: AstrMessageEvent):
        async for result in self._process_generation_request(event, "cos自拍", require_image=True): yield result
    @filter.command("自定义图生图", prefix_optional=True)
    async def on_cmd_img_to_img(self, event: AstrMessageEvent):
        async for result in self._process_generation_request(event, "自定义图生图", require_image=True): yield result
    @filter.command("自定义文生图", prefix_optional=True)
    async def on_cmd_text_to_image(self, event: AstrMessageEvent):
        async for result in self._process_generation_request(event, "自定义文生图", require_image=False): yield result
    @filter.command("孤独的我", prefix_optional=True)
    async def on_cmd_clown(self, event: AstrMessageEvent):
        async for result in self._process_generation_request(event, "孤独的我", require_image=True): yield result
    @filter.command("第三视角", prefix_optional=True)
    async def on_cmd_view3(self, event: AstrMessageEvent):
        async for result in self._process_generation_request(event, "第三视角", require_image=True): yield result
    @filter.command("鬼图", prefix_optional=True)
    async def on_cmd_ghost(self, event: AstrMessageEvent):
        async for result in self._process_generation_request(event, "鬼图", require_image=True): yield result
    @filter.command("第一视角", prefix_optional=True)
    async def on_cmd_view1(self, event: AstrMessageEvent):
        async for result in self._process_generation_request(event, "第一视角", require_image=True): yield result
    @filter.command("贴纸化", prefix_optional=True)
    async def on_cmd_sticker(self, event: AstrMessageEvent):
        async for result in self._process_generation_request(event, "贴纸化", require_image=True): yield result
    @filter.command("玉足", prefix_optional=True)
    async def on_cmd_foot_jade(self, event: AstrMessageEvent):
        async for result in self._process_generation_request(event, "玉足", require_image=True): yield result
    @filter.command("fumo化", prefix_optional=True)
    async def on_cmd_fumo(self, event: AstrMessageEvent):
        async for result in self._process_generation_request(event, "fumo化", require_image=True): yield result
    @filter.command("生图帮助", prefix_optional=True)
    async def on_cmd_help(self, event: AstrMessageEvent):
        async for result in self._process_generation_request(event, "生图帮助", require_image=False): yield result

    # ------------------- 核心处理逻辑 -------------------

    async def _process_generation_request(self, event: AstrMessageEvent, cmd: str, require_image: bool, natural_prompt: str = "", pre_fetched_images: Optional[List[bytes]] = None):
        cmd_text = event.message_str
        cmd_map = {"手办化": "figurine_1", "手办化2": "figurine_2", "手办化3": "figurine_3", "手办化4": "figurine_4",
                   "手办化5": "figurine_5", "手办化6": "figurine_6", "Q版化": "q_version", "痛屋化": "pain_room_1",
                   "痛屋化2": "pain_room_2", "痛车化": "pain_car", "cos化": "cos", "cos自拍": "cos_selfie",
                   "孤独的我": "clown", "第三视角": "view_3", "鬼图": "ghost", "第一视角": "view_1", "贴纸化": "sticker",
                   "玉足": "foot_jade", "fumo化": "fumo"}

        if cmd == "生图帮助":
            help_text = self.conf.get("help_text", "帮助信息未配置。")
            yield event.plain_result(help_text)
            return

        user_prompt = ""
        if cmd in ["自定义图生图", "自定义文生图"]:
            user_prompt = cmd_text.strip()
            if not user_prompt:
                error_msg = "❌ 命令格式错误: /自定义图生图 <提示词> [图片]" if cmd == "自定义图生图" else "❌ 命令格式错误: /自定义文生图 <提示词>"
                yield event.plain_result(error_msg)
                return
        elif cmd.startswith("自然语言"):
            user_prompt = natural_prompt
        else:
            prompt_key = cmd_map.get(cmd)
            user_prompts = self.conf.get("prompts", {})
            user_prompt = user_prompts.get(prompt_key) or self.default_prompts.get(prompt_key, "")
            if not user_prompt:
                yield event.plain_result(f"❌ 预设 '{cmd}' 未在配置中找到或prompt为空。")
                return

        sender_id, group_id, is_master = event.get_sender_id(), event.get_group_id(), self.is_global_admin(event)
        if not is_master:
            user_count = self._get_user_count(sender_id)
            group_count = self._get_group_count(group_id) if group_id else 0
            user_limit_on = self.conf.get("enable_user_limit", True)
            group_limit_on = self.conf.get("enable_group_limit", False) and group_id
            
            has_count = (not group_limit_on or group_count > 0) or (not user_limit_on or user_count > 0)
            if group_id and not has_count:
                yield event.plain_result("❌ 本群次数与您的个人次数均已用尽。"); return
            if not group_id and user_limit_on and user_count <= 0:
                yield event.plain_result("❌ 您的使用次数已用完。"); return

        img_bytes_list = []
        if require_image:
            # 优先使用预加载的图片（来自LLM工具）
            if pre_fetched_images is not None:
                img_bytes_list = pre_fetched_images
            # 否则，执行指令的图片查找逻辑（包含头像）
            elif not self.iwf or not (img_bytes_list := await self.iwf.get_all_images(event)):
                yield event.plain_result("此命令需要图片。请发送或引用一张图片，或@一个用户再试。"); return
            
            yield event.plain_result(f"🎨 收到 {len(img_bytes_list)} 张图片，正在生成 [{cmd}] 风格的图片...")
        else:
            yield event.plain_result(f"🎨 收到指令，正在生成 [{cmd}] 风格的图片...")

        start_time = datetime.now()
        res = await self._call_api_with_retry(img_bytes_list, user_prompt)
        elapsed = (datetime.now() - start_time).total_seconds()

        if isinstance(res, bytes):
            if not is_master:
                if self.conf.get("enable_group_limit", False) and group_id and self._get_group_count(group_id) > 0:
                    await self._decrease_group_count(group_id)
                elif self.conf.get("enable_user_limit", True) and self._get_user_count(sender_id) > 0:
                    await self._decrease_user_count(sender_id)

            caption_parts = [f"✅ 生成成功 ({elapsed:.2f}s)", f"模式: {cmd}"]
            if is_master:
                caption_parts.append("剩余次数: ∞")
            else:
                if self.conf.get("enable_user_limit", True): caption_parts.append(f"个人剩余: {self._get_user_count(sender_id)}")
                if self.conf.get("enable_group_limit", False) and group_id: caption_parts.append(f"本群剩余: {self._get_group_count(group_id)}")
            yield event.chain_result([Image.fromBytes(res), Plain(" | ".join(caption_parts))])
        else:
            yield event.plain_result(f"❌ 生成失败 ({elapsed:.2f}s)\n原因: {res}")

    async def _get_current_api_key(self) -> str | None:
        keys = self.conf.get("api_keys", [])
        if not keys: return None
        async with self.key_lock:
            return keys[self.key_index]

    async def _switch_next_api_key(self):
        keys = self.conf.get("api_keys", [])
        if not keys: return
        async with self.key_lock:
            self.key_index = (self.key_index + 1) % len(keys)
            logger.info(f"API密钥已切换至索引: {self.key_index}")

    def _extract_image_url_from_response(self, data: Dict[str, Any]) -> str | None:
        try: return data["choices"][0]["message"]["images"][0]["image_url"]["url"]
        except (IndexError, TypeError, KeyError): pass
        try: return data["choices"][0]["message"]["images"][0]["url"]
        except (IndexError, TypeError, KeyError): pass
        try:
            content_text = data["choices"][0]["message"]["content"]
            url_match = re.search(r'https?://[^\s<>")\]]+', content_text)
            if url_match: return url_match.group(0).rstrip(")>,'\"")
        except (IndexError, TypeError, KeyError): pass
        return None

    async def _call_api_with_retry(self, image_bytes_list: List[bytes], prompt: str) -> bytes | str:
        api_keys = self.conf.get("api_keys", [])
        if not api_keys:
            return "无可用的 API Key"

        max_attempts = len(api_keys)
        for attempt in range(max_attempts):
            api_key = await self._get_current_api_key()
            if not api_key:
                continue

            logger.info(f"尝试使用API密钥 (索引: {self.key_index}, 尝试次数: {attempt + 1}/{max_attempts}) 进行生图...")
            
            try:
                result = await self._call_api_single(api_key, image_bytes_list, prompt)
                return result
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.error(f"第 {attempt + 1} 次尝试失败 (密钥索引 {self.key_index}): 网络错误 {e}")
                await self._switch_next_api_key()
            except Exception as e:
                logger.error(f"第 {attempt + 1} 次尝试失败 (密钥索引 {self.key_index}): 未知错误 {e}", exc_info=True)
                await self._switch_next_api_key()
        
        return "所有API密钥均尝试失败，请检查密钥配置或网络连接。"

    async def _call_api_single(self, api_key: str, image_bytes_list: List[bytes], prompt: str) -> bytes | str:
        api_url = self.conf.get("api_url")
        if not api_url: return "API URL 未配置"

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        content_list = [{"type": "text", "text": prompt}]
        if image_bytes_list:
            for image_bytes in image_bytes_list:
                img_b64 = base64.b64encode(image_bytes).decode("utf-8")
                content_list.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})

        payload = {"model": "nano-banana", "max_tokens": 1500, "stream": False, "messages": [{"role": "user", "content": content_list}]}
        
        if not self.iwf: return "ImageWorkflow 未初始化"
        async with self.iwf.session.post(api_url, json=payload, headers=headers, proxy=self.iwf.proxy, timeout=120) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                logger.error(f"API 请求失败: HTTP {resp.status}, 响应: {error_text}")
                raise aiohttp.ClientResponseError(resp.request_info, resp.history, status=resp.status, message=error_text)
            
            data = await resp.json()
            if "error" in data:
                return data["error"].get("message", json.dumps(data["error"]))
            
            gen_image_url = self._extract_image_url_from_response(data)
            if not gen_image_url:
                error_msg = f"API响应中未找到图片数据。原始响应 (部分): {str(data)[:500]}..."
                logger.error(f"API响应中未找到图片数据: {data}")
                return error_msg
            
            if gen_image_url.startswith("data:image/"):
                return base64.b64decode(gen_image_url.split(",", 1)[1])
            else:
                downloaded_image = await self.iwf._download_image(gen_image_url)
                if downloaded_image:
                    return downloaded_image
                else:
                    raise Exception("下载生成的图片失败")

    async def terminate(self):
        if self.iwf: await self.iwf.terminate()
        logger.info("[NanoBananaPlugin] 插件已终止")
