from typing import ClassVar, Mapping, NoReturn


class PlaceholderModule:
    module_id: ClassVar[str]

    def evaluate(self, timestamp_ns: int, inputs: Mapping[str, object]) -> NoReturn:
        raise NotImplementedError(
            f"{self.module_id} is a PLACEHOLDER; no simulated outputs are available."
        )
