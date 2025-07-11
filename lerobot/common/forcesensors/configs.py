# forcesensorconfig的base class

import abc
from dataclasses import dataclass

import draccus


@dataclass(kw_only=True)
class ForceSensorConfig(draccus.ChoiceRegistry, abc.ABC):
    # 这个类也没什么可以定义的……



    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

