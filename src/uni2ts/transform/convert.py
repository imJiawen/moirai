from ._base import Transformation
from typing import Any, Optional
import numpy as np

@dataclass
class toArray(Transformation):
    output_field: str
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    feat: bool = False

    def __post_init__(self):
        self.pack_str: str = "* feat" if self.feat else "*"

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        fields = self.collect_func_list(
            self.pop_field,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        if len(fields) > 0:
            output_field = fields[0]
            data_entry |= {self.output_field: output_field}
        return data_entry

    @staticmethod
    def pop_field(data_entry: dict[str, Any], field: str) -> Any:
        return np.asarray(data_entry.pop(field))

