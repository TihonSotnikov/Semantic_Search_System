import numpy as np

from typing import Any
from sqlalchemy import Dialect
from torch import Tensor, from_numpy
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator, LargeBinary


class Base(DeclarativeBase):
    pass


class TensorType(TypeDecorator):
    impl = LargeBinary

    def process_bind_param(self, value: Tensor | None, dialect: Dialect) -> Any:
        if value is not None:
            return value.detach().cpu().numpy().astype(np.float32).tobytes()
        return None
    
    def process_result_value(self, value: bytes | None, dialect: Dialect) -> Tensor | None:
        if value is not None:
            arr = np.frombuffer(value, dtype=np.float32).copy()
            return from_numpy(arr)
        return None


class Knowledge(Base):
    __tablename__ = "Knowledge"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column()
    text: Mapped[str] = mapped_column()
    vector: Mapped[Tensor] = mapped_column(TensorType)
