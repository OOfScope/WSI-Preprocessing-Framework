
from abc import ABC, abstractmethod


class LabelingStrategy(ABC):
    @abstractmethod
    def annotate(self, mask):
        pass

class TopKLabeling(LabelingStrategy):
    def annotate(self, mask):
        print(f"TopKLabeling.")

class TopKThrLabeling(LabelingStrategy):
    def annotate(self, mask):
        print(f"TopKThrLabeling.")
        
        
        
        
class LabelingContext:
    def __init__(self, labeling_strategy:LabelingStrategy):
        self.labeling_strategy = labeling_strategy

    def set_labeling_strategy(self, labeling_strategy):
        self.labeling_strategy = labeling_strategy

    def process_annotation(self, mask):
        self.labeling_strategy.annotate(mask)
        
        