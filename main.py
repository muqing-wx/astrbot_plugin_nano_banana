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
from PIL import Image as PILImage, ImageDraw, ImageFont

from astrbot import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import At, Image, Reply, Plain
from astrbot.core.platform.astr_message_event import AstrMessageEvent


@register(
    "astrbot_plugin_nano_banana",
    "沐沐沐倾",
    "一款功能强大的AI生图插件，基于柏拉图API，集成了多种预设风格、智能统一指令、及后台管理功能。",
    "1.0.7", # 采用全新的专业级帮助图排版
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
                        if isinstance(s_chain, Image): await process_image(s_chain)
            for seg in event.message_obj.message:
                if isinstance(seg, Image): await process_image(seg)
            return images

        async def get_explicit_images_only(self, event: AstrMessageEvent) -> List[bytes]:
            return await self._get_images_from_segments(event)

        async def get_all_images_for_preset_cmd(self, event: AstrMessageEvent) -> List[bytes]:
            if images := await self._get_images_from_segments(event): return images
            if at_user_id := next((str(s.qq) for s in event.message_obj.message if isinstance(s, At)), None):
                if avatar := await self._get_avatar(at_user_id): return [avatar]
            return []

        async def terminate(self):
            if self.session and not self.session.closed: await self.session.close()

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
        self.font_path = Path(__file__).parent / "resources" / "font.ttf"
        self.fonts = {}
        self.promo_text = "柏拉图AI_API中转站: "
        self.promo_link = "https://api.bltcy.ai/register?aff=63Ig"

    async def initialize(self):
        prompts_file = Path(__file__).parent / "prompts.json"
        if prompts_file.exists():
            try: self.default_prompts = json.loads(prompts_file.read_text("utf-8"))
            except Exception as e: logger.error(f"加载 prompts.json 失败: {e}", exc_info=True)
        
        if self.font_path.exists():
            try:
                self.fonts['title'] = ImageFont.truetype(str(self.font_path), 52)
                self.fonts['header'] = ImageFont.truetype(str(self.font_path), 34)
                self.fonts['body'] = ImageFont.truetype(str(self.font_path), 26)
                logger.info(f"帮助图片字体已加载: {self.font_path}")
            except Exception as e: logger.warning(f"加载字体失败，帮助信息将以文本发送: {e}")
        else: logger.warning(f"字体文件未找到: {self.font_path}。帮助信息将以文本发送。")

        proxy_url = self.conf.get("proxy_url") if self.conf.get("use_proxy", False) else None
        self.iwf = self.ImageWorkflow(proxy_url)
        await self._load_user_counts()
        await self._load_group_counts()
        logger.info("Nano Banana 生图插件已加载")
        if not self.conf.get("api_keys"): logger.warning("NanoBananaPlugin: 未配置任何API密钥")

    async def uninstall(self):
        logger.info("正在卸载 Nano Banana 插件...")
        try:
            if self.user_counts_file.exists(): self.user_counts_file.unlink()
            if self.group_counts_file.exists(): self.group_counts_file.unlink()
        except Exception as e: logger.error(f"卸载插件时清理文件失败: {e}", exc_info=True)

    def is_global_admin(self, event: AstrMessageEvent) -> bool:
        return event.get_sender_id() in self.context.get_config().get("admins_id", [])

    def _render_text_to_image_sync(self, text: str) -> bytes | None:
        if not self.fonts: return None

        PADDING = 60
        TITLE_SPACING = 40
        SECTION_SPACING = 25
        LINE_SPACING = 18
        BG_COLOR = (240, 240, 245)
        TITLE_COLOR = (20, 20, 20)
        HEADER_COLOR = (58, 77, 143)
        BODY_COLOR = (51, 51, 51)
        LINE_COLOR = (220, 220, 225)

        lines = text.strip().split('\n')
        
        content_blocks = []
        max_width = 0
        for line in lines:
            line = line.strip()
            font, content = None, ""
            if line.startswith('# '): font, content = self.fonts['title'], line[2:]
            elif line.startswith('## '): font, content = self.fonts['header'], line[3:]
            elif line.startswith('* '): font, content = self.fonts['body'], line[2:]
            elif line.startswith('---'): font, content = None, '---'
            elif line: font, content = self.fonts['body'], line
            
            if font:
                width = font.getbbox(content)[2]
                if line.startswith('* '): width += 40
                if width > max_width: max_width = width
            content_blocks.append({'type': line[:3] if line else 'empty', 'content': content, 'font': font})

        total_height = PADDING
        for block in content_blocks:
            if block['type'] == '#  ': total_height += block['font'].getbbox(block['content'])[3] + TITLE_SPACING
            elif block['type'] == '## ': total_height += block['font'].getbbox(block['content'])[3] + SECTION_SPACING
            elif block['type'] == '*  ': total_height += block['font'].getbbox(block['content'])[3] + LINE_SPACING
            elif block['type'] == '---': total_height += 30
            elif block['type'] == 'empty': total_height += LINE_SPACING
            else: total_height += block['font'].getbbox(block['content'])[3] + LINE_SPACING
        total_height += PADDING - LINE_SPACING

        img_width = max_width + PADDING * 2
        image = PILImage.new('RGB', (img_width, total_height), BG_COLOR)
        draw = ImageDraw.Draw(image)
        
        y = PADDING
        for block in content_blocks:
            if block['type'] == '#  ':
                draw.text((PADDING, y), block['content'], font=block['font'], fill=TITLE_COLOR)
                y += block['font'].getbbox(block['content'])[3] + TITLE_SPACING
            elif block['type'] == '## ':
                draw.text((PADDING, y), block['content'], font=block['font'], fill=HEADER_COLOR)
                y += block['font'].getbbox(block['content'])[3] + SECTION_SPACING
            elif block['type'] == '*  ':
                text_height = block['font'].getbbox(block['content'])[3] - block['font'].getbbox(block['content'])[1]
                bullet_radius = 4
                bullet_y = y + text_height / 2
                draw.ellipse((PADDING, bullet_y - bullet_radius, PADDING + bullet_radius*2, bullet_y + bullet_radius), fill=HEADER_COLOR)
                draw.text((PADDING + 40, y), block['content'], font=block['font'], fill=BODY_COLOR)
                y += text_height + LINE_SPACING
            elif block['type'] == '---':
                draw.line([(PADDING, y+10), (img_width - PADDING, y+10)], fill=LINE_COLOR, width=2)
                y += 30
            elif block['type'] == 'empty':
                y += LINE_SPACING
            else:
                draw.text((PADDING, y), block['content'], font=block['font'], fill=BODY_COLOR)
                y += block['font'].getbbox(block['content'])[3] + LINE_SPACING

        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return buffer.getvalue()

    async def _load_user_counts(self):
        if not self.user_counts_file.exists(): self.user_counts = {}; return
        try: self.user_counts = {str(k): v for k, v in json.loads(self.user_counts_file.read_text("utf-8")).items()}
        except Exception as e: logger.error(f"加载用户次数文件失败: {e}", exc_info=True); self.user_counts = {}

    async def _save_user_counts(self):
        try: self.user_counts_file.write_text(json.dumps(self.user_counts, ensure_ascii=False, indent=4), "utf-8")
        except Exception as e: logger.error(f"保存用户次数文件失败: {e}", exc_info=True)

    def _get_user_count(self, user_id: str) -> int: return self.user_counts.get(str(user_id), 0)

    async def _decrease_user_count(self, user_id: str):
        if (count := self._get_user_count(str(user_id))) > 0:
            self.user_counts[str(user_id)] = count - 1; await self._save_user_counts()

    async def _load_group_counts(self):
        if not self.group_counts_file.exists(): self.group_counts = {}; return
        try: self.group_counts = {str(k): v for k, v in json.loads(self.group_counts_file.read_text("utf-8")).items()}
        except Exception as e: logger.error(f"加载群组次数文件失败: {e}", exc_info=True); self.group_counts = {}

    async def _save_group_counts(self):
        try: self.group_counts_file.write_text(json.dumps(self.group_counts, ensure_ascii=False, indent=4), "utf-8")
        except Exception as e: logger.error(f"保存群组次数文件失败: {e}", exc_info=True)

    def _get_group_count(self, group_id: str) -> int: return self.group_counts.get(str(group_id), 0)

    async def _decrease_group_count(self, group_id: str):
        if (count := self._get_group_count(str(group_id))) > 0:
            self.group_counts[str(group_id)] = count - 1; await self._save_group_counts()

    @filter.command("生图", "draw", "画画", prefix_optional=True)
    async def on_cmd_draw(self, event: AstrMessageEvent):
        if not self.iwf: yield event.plain_result("插件内部错误：ImageWorkflow未初始化。"); return
        images = await self.iwf.get_explicit_images_only(event)
        mode = "图生图" if images else "文生图"
        async for result in self._process_generation_request(event, mode=mode, require_image=bool(images), pre_fetched_images=images): yield result

    @filter.command("生图增加用户次数", prefix_optional=True)
    async def on_add_user_counts(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        cmd_text = event.message_str.strip()
        target_qq, count = None, 0
        if at_seg := next((s for s in event.message_obj.message if isinstance(s, At)), None):
            target_qq = str(at_seg.qq)
            if match := re.search(r"(\d+)\s*$", cmd_text): count = int(match.group(1))
        elif match := re.search(r"(\d+)\s+(\d+)", cmd_text): target_qq, count = match.group(1), int(match.group(2))
        if not target_qq or count <= 0: yield event.plain_result('格式错误:\n/生图增加用户次数 @用户 <次数>\n或 /生图增加用户次数 <QQ号> <次数>'); return
        current_count = self._get_user_count(target_qq)
        self.user_counts[str(target_qq)] = current_count + count
        await self._save_user_counts()
        yield event.plain_result(f"✅ 已为用户 {target_qq} 增加 {count} 次，TA当前剩余 {current_count + count} 次。")

    @filter.command("生图增加群组次数", prefix_optional=True)
    async def on_add_group_counts(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        if not (match := re.search(r"(\d+)\s+(\d+)", event.message_str.strip())): yield event.plain_result('格式错误: /生图增加群组次数 <群号> <次数>'); return
        target_group, count = match.group(1), int(match.group(2))
        current_count = self._get_group_count(target_group)
        self.group_counts[str(target_group)] = current_count + count
        await self._save_group_counts()
        yield event.plain_result(f"✅ 已为群组 {target_group} 增加 {count} 次，该群当前剩余 {current_count + count} 次。")

    @filter.command("生图查询次数", prefix_optional=True)
    async def on_query_counts(self, event: AstrMessageEvent):
        user_id_to_query = event.get_sender_id()
        if self.is_global_admin(event):
            if at_seg := next((s for s in event.message_obj.message if isinstance(s, At)), None): user_id_to_query = str(at_seg.qq)
            elif match := re.search(r"(\d+)", event.message_str): user_id_to_query = match.group(1)
        user_count = self._get_user_count(user_id_to_query)
        reply_msg = f"用户 {user_id_to_query} 个人剩余次数: {user_count}" if user_id_to_query != event.get_sender_id() else f"您好，您当前个人剩余次数: {user_count}"
        if group_id := event.get_group_id(): reply_msg += f"\n本群共享剩余次数: {self._get_group_count(group_id)}"
        yield event.plain_result(reply_msg)

    @filter.command("生图添加key", prefix_optional=True)
    async def on_add_key(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        if not (new_keys := event.message_str.strip().split()): yield event.plain_result("格式错误，请提供要添加的Key。"); return
        api_keys = self.conf.get("api_keys", [])
        added_keys = [key for key in new_keys if key not in api_keys]
        api_keys.extend(added_keys)
        await self.conf.set("api_keys", api_keys)
        yield event.plain_result(f"✅ 操作完成，新增 {len(added_keys)} 个Key，当前共 {len(api_keys)} 个。")

    @filter.command("生图key列表", prefix_optional=True)
    async def on_list_keys(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        api_keys = self.conf.get("api_keys", [])
        if not api_keys: yield event.plain_result("📝 暂未配置任何 API Key。"); return
        key_list_str = "\n".join(f"{i + 1}. {key[:8]}...{key[-4:]}" for i, key in enumerate(api_keys))
        yield event.plain_result(f"🔑 API Key 列表:\n{key_list_str}")

    @filter.command("生图删除key", prefix_optional=True)
    async def on_delete_key(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        param = event.message_str.strip()
        api_keys = self.conf.get("api_keys", [])
        if param.lower() == "all":
            await self.conf.set("api_keys", []); yield event.plain_result(f"✅ 已删除全部 {len(api_keys)} 个 Key。")
        elif param.isdigit() and 1 <= int(param) <= len(api_keys):
            removed_key = api_keys.pop(int(param) - 1)
            await self.conf.set("api_keys", api_keys); yield event.plain_result(f"✅ 已删除 Key: {removed_key[:8]}...")
        else: yield event.plain_result("格式错误，请使用 /生图删除key <序号|all>")

    PRESET_COMMANDS = ["手办化", "手办化2", "手办化3", "手办化4", "手办化5", "手办化6", "Q版化", "痛屋化", "痛屋化2", "痛车化", "cos化", "cos自拍", "孤独的我", "第三视角", "鬼图", "第一视角", "贴纸化", "玉足", "fumo化"]
    for cmd in PRESET_COMMANDS:
        exec(f"""
@filter.command("{cmd}", prefix_optional=True)
async def on_cmd_{cmd}(self, event: AstrMessageEvent):
    async for result in self._process_generation_request(event, mode="{cmd}", require_image=True): yield result
""")
    @filter.command("生图帮助", prefix_optional=True)
    async def on_cmd_help(self, event: AstrMessageEvent):
        async for result in self._process_generation_request(event, mode="生图帮助", require_image=False): yield result

    async def _process_generation_request(self, event: AstrMessageEvent, mode: str, require_image: bool, pre_fetched_images: Optional[List[bytes]] = None):
        if mode == "生图帮助":
            help_text = self.conf.get("help_text", "帮助信息未配置。")
            promo_message = Plain(f"\n{self.promo_text}{self.promo_link}")
            loop = asyncio.get_running_loop()
            image_bytes = await loop.run_in_executor(None, self._render_text_to_image_sync, help_text)
            if image_bytes: yield event.chain_result([Image.fromBytes(image_bytes), promo_message])
            else: yield event.chain_result([Plain(help_text), promo_message])
            return

        user_prompt = ""
        if mode in ["文生图", "图生图"]:
            if not (user_prompt := event.message_str.strip()): yield event.plain_result(f"❌ 命令格式错误: /{event.command} <提示词> [图片]"); return
        else:
            cmd_map = {"手办化": "figurine_1", "手办化2": "figurine_2", "手办化3": "figurine_3", "手办化4": "figurine_4", "手办化5": "figurine_5", "手办化6": "figurine_6", "Q版化": "q_version", "痛屋化": "pain_room_1", "痛屋化2": "pain_room_2", "痛车化": "pain_car", "cos化": "cos", "cos自拍": "cos_selfie", "孤独的我": "clown", "第三视角": "view_3", "鬼图": "ghost", "第一视角": "view_1", "贴纸化": "sticker", "玉足": "foot_jade", "fumo化": "fumo"}
            prompt_key = cmd_map.get(mode)
            user_prompts = self.conf.get("prompts", {})
            if not (user_prompt := user_prompts.get(prompt_key) or self.default_prompts.get(prompt_key, "")):
                yield event.plain_result(f"❌ 预设 '{mode}' 未在配置中找到或prompt为空。"); return

        sender_id, group_id, is_master = event.get_sender_id(), event.get_group_id(), self.is_global_admin(event)
        if not is_master:
            user_limit_on = self.conf.get("enable_user_limit", True)
            group_limit_on = self.conf.get("enable_group_limit", False) and group_id
            if not ((not group_limit_on or self._get_group_count(group_id) > 0) or (not user_limit_on or self._get_user_count(sender_id) > 0)):
                yield event.plain_result("❌ 本群次数与您的个人次数均已用尽。"); return

        if require_image:
            img_bytes_list = pre_fetched_images if pre_fetched_images is not None else (await self.iwf.get_all_images_for_preset_cmd(event) if self.iwf else [])
            if not img_bytes_list: yield event.plain_result("此命令需要图片。请发送或引用一张图片，或@一个用户再试。"); return
            yield event.plain_result(f"🎨 收到 {len(img_bytes_list)} 张图片，正在生成 [{mode}] ...")
        else:
            yield event.plain_result(f"🎨 收到指令，正在生成 [{mode}] ...")

        start_time = datetime.now()
        res = await self._call_api_with_retry(img_bytes_list if require_image else [], user_prompt)
        elapsed = (datetime.now() - start_time).total_seconds()

        if isinstance(res, bytes):
            if not is_master:
                if self.conf.get("enable_group_limit", False) and group_id and self._get_group_count(group_id) > 0: await self._decrease_group_count(group_id)
                elif self.conf.get("enable_user_limit", True) and self._get_user_count(sender_id) > 0: await self._decrease_user_count(sender_id)
            caption_parts = [f"✅ 生成成功 ({elapsed:.2f}s)", f"模式: {mode}"]
            if is_master: caption_parts.append("剩余: ∞")
            else:
                if self.conf.get("enable_user_limit", True): caption_parts.append(f"个人剩余: {self._get_user_count(sender_id)}")
                if self.conf.get("enable_group_limit", False) and group_id: caption_parts.append(f"本群剩余: {self._get_group_count(group_id)}")
            yield event.chain_result([Image.fromBytes(res), Plain(" | ".join(caption_parts))])
        else:
            yield event.plain_result(f"❌ 生成失败 ({elapsed:.2f}s)\n原因: {res}")

    async def _get_current_api_key(self) -> str | None:
        keys = self.conf.get("api_keys", [])
        if not keys: return None
        async with self.key_lock: return keys[self.key_index]

    async def _switch_next_api_key(self):
        keys = self.conf.get("api_keys", [])
        if not keys: return
        async with self.key_lock: self.key_index = (self.key_index + 1) % len(keys)
        logger.info(f"API密钥已切换至索引: {self.key_index}")

    def _extract_image_url_from_response(self, data: Dict[str, Any]) -> str | None:
        try: return data["choices"][0]["message"]["images"][0]["image_url"]["url"]
        except (IndexError, TypeError, KeyError): pass
        try: return data["choices"][0]["message"]["images"][0]["url"]
        except (IndexError, TypeError, KeyError): pass
        try:
            if url_match := re.search(r'https?://[^\s<>")\]]+', data["choices"][0]["message"]["content"]): return url_match.group(0).rstrip(")>,'\"")
        except (IndexError, TypeError, KeyError): pass
        return None

    async def _call_api_with_retry(self, image_bytes_list: List[bytes], prompt: str) -> bytes | str:
        api_keys = self.conf.get("api_keys", [])
        if not api_keys: return "无可用的 API Key"
        for attempt in range(len(api_keys)):
            if not (api_key := await self._get_current_api_key()): continue
            logger.info(f"尝试使用API密钥 (索引: {self.key_index}, 尝试: {attempt + 1}/{len(api_keys)}) 进行生图...")
            try: return await self._call_api_single(api_key, image_bytes_list, prompt)
            except Exception as e:
                logger.error(f"尝试失败 (密钥索引 {self.key_index}): {e}", exc_info=False)
                await self._switch_next_api_key()
        return "所有API密钥均尝试失败，请检查密钥配置或网络连接。"

    async def _call_api_single(self, api_key: str, image_bytes_list: List[bytes], prompt: str) -> bytes | str:
        if not (api_url := self.conf.get("api_url")): return "API URL 未配置"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        content_list = [{"type": "text", "text": prompt}]
        for image_bytes in image_bytes_list:
            content_list.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"}})
        payload = {"model": "nano-banana", "max_tokens": 1500, "stream": False, "messages": [{"role": "user", "content": content_list}]}
        
        if not self.iwf: return "ImageWorkflow 未初始化"
        async with self.iwf.session.post(api_url, json=payload, headers=headers, proxy=self.iwf.proxy, timeout=120) as resp:
            resp.raise_for_status()
            data = await resp.json()
            if "error" in data: return data["error"].get("message", json.dumps(data["error"]))
            if not (gen_image_url := self._extract_image_url_from_response(data)):
                raise Exception(f"API响应中未找到图片数据: {str(data)[:500]}...")
            if gen_image_url.startswith("data:image/"): return base64.b64decode(gen_image_url.split(",", 1)[1])
            if downloaded_image := await self.iwf._download_image(gen_image_url): return downloaded_image
            raise Exception("下载生成的图片失败")

    async def terminate(self):
        if self.iwf: await self.iwf.terminate()
        logger.info("[NanoBananaPlugin] 插件已终止")
