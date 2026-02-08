"""Defines a base trainer mixin for handling subprocess monitoring jobs."""

import logging
import multiprocessing as mp
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from multiprocessing.managers import SyncManager
from typing import Generic, TypeVar

import jax

from xax.task.base import BaseConfig, BaseTask
from xax.utils.structured_config import field

logger: logging.Logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class ProcessConfig(BaseConfig):
    multiprocessing_context: str | None = field("spawn", help="The multiprocessing context to use")
    disable_multiprocessing: bool = field(False, help="If set, disable multiprocessing")


Config = TypeVar("Config", bound=ProcessConfig)


class ProcessMixin(BaseTask[Config], Generic[Config]):
    """Defines a base trainer mixin for handling monitoring processes."""

    _mp_ctx: BaseContext | None
    _mp_manager: SyncManager | None

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        if self.config.disable_multiprocessing:
            self._mp_ctx = None
            self._mp_manager = None
        else:
            self._mp_ctx = mp.get_context(config.multiprocessing_context)
            self._mp_manager = self._mp_ctx.Manager()

    @property
    def multiprocessing_context(self) -> BaseContext | None:
        return self._mp_ctx

    @property
    def multiprocessing_manager(self) -> SyncManager | None:
        return self._mp_manager

    def on_training_end(self) -> None:
        super().on_training_end()

        if self._mp_manager is not None:
            self._mp_manager.shutdown()
            self._mp_manager.join()
