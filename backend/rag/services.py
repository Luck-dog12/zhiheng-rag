from rag.providers import RemoteModels
from rag.retrieval import Retriever
from rag.storage import S3Storage
from rag.vector import MilvusIndex


class Services:
    def __init__(self, settings, *, models=None, storage=None, index=None):
        self.models = models or RemoteModels(settings)
        self.storage = storage or S3Storage(settings)
        self.index = index or MilvusIndex(settings)
        self.retriever = Retriever(settings, self.models, self.index)

    async def close(self):
        for service in (self.models, self.storage, self.index):
            await service.close()
