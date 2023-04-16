""" Define a way of visualising various items (plotter, essentially) """

from typing import Type

from planetary_system.system import _system_


class SystemVisualiser(_system_):
    def __init__(self, System: Type[_system_]) -> None:
        super().__init__(f"{System._name}.visualiser")
        self.system = System
        self.logger.info("Attached visualiser")

    def plot_system(self, style: str = "simple") -> None:
        match style.lower():
            case "simple":
                return self.plot_simple()
        raise Exception(f"Unknown plotting style {style}")

    def plot_simple(self) -> None:
        # for body in self.flattened:
        #     print(self.find_parent(body))
        pass
