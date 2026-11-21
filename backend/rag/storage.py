import asyncio
import base64
import hashlib
from functools import cached_property

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from rag.config import Settings
from rag.errors import AppError


def _delete_content_md5(params, **kwargs):
    # COS requires Content-MD5 over the serialized XML; newer botocore uses CRC32 by default.
    # This header is also part of the standard S3 multi-object deletion contract.
    params["headers"]["Content-MD5"] = base64.b64encode(
        hashlib.md5(params["body"], usedforsecurity=False).digest()).decode("ascii")


class S3Storage:
    def __init__(self, settings: Settings):
        self.settings = settings

    def configured(self):
        s = self.settings
        if not all((s.s3_endpoint_url, s.s3_bucket, s.s3_access_key.get_secret_value(), s.s3_secret_key.get_secret_value())):
            raise AppError("missing_configuration", "缺少 S3 对象存储配置", 503)

    @cached_property
    def client(self):
        self.configured()
        s = self.settings
        client = boto3.client("s3", endpoint_url=s.s3_endpoint_url, region_name=s.s3_region,
                            aws_access_key_id=s.s3_access_key.get_secret_value(),
                            aws_secret_access_key=s.s3_secret_key.get_secret_value(),
                            config=Config(signature_version=s.s3_signature_version,
                                          s3={"addressing_style": s.s3_addressing_style},
                                          connect_timeout=10, read_timeout=60,
                                          retries={"total_max_attempts": 3, "mode": "standard"}))
        client.meta.events.register("before-call.s3.DeleteObjects", _delete_content_md5)
        return client

    async def _call(self, fn, **kwargs):
        try:
            return await asyncio.to_thread(fn, **kwargs)
        except (BotoCoreError, ClientError):
            raise AppError("storage_unavailable", "对象存储操作失败，请检查服务连接及权限", 502) from None

    async def put(self, key: str, data: bytes, content_type: str = "application/octet-stream"):
        await self._call(self.client.put_object, Bucket=self.settings.s3_bucket, Key=key,
                         Body=data, ContentType=content_type)

    async def get(self, key: str) -> bytes:
        result = await self._call(self.client.get_object, Bucket=self.settings.s3_bucket, Key=key)
        try:
            return await self._call(result["Body"].read)
        finally:
            result["Body"].close()

    async def delete(self, keys: list[str]):
        for start in range(0, len(keys), 1000):
            result = await self._call(self.client.delete_objects, Bucket=self.settings.s3_bucket,
                                     Delete={"Objects": [{"Key": key} for key in keys[start:start + 1000]]})
            if result.get("Errors"):
                raise AppError("storage_delete_failed", "部分文件删除失败，稍后将重试", 502)

    async def check(self):
        await self._call(self.client.head_bucket, Bucket=self.settings.s3_bucket)

    async def close(self):
        if "client" in self.__dict__:
            await asyncio.to_thread(self.client.close)
