
from abc import ABC, abstractmethod


class LabelingStrategy(ABC):
    @abstractmethod
    def annotate(self):
        pass
    
    @abstractmethod
    def get_strategy_signature(self):
        pass
    
    def set_perc_dict(self, perc_dict):
        self.perc_dict = perc_dict


class TopKLabeling(LabelingStrategy):
    def __init__(self, k:int):
        self.k = k
        print(f"Built TopKLabeling strategy, k={k}")
    
    def annotate(self):
        return sorted(self.perc_dict.items(), key=lambda x: x[1], reverse=True)[:self.k]
    
    def get_strategy_signature(self):
        return f"TopKLabeling_k{self.k}"


class TopKThrLabeling(LabelingStrategy):
    def __init__(self, k:int, thr:float):
        self.k = k
        self.thr = thr
        
    def annotate(self):
        return [(k, v) for k, v in self.perc_dict.items() if v >= self.thr][:self.k]        
        
    def get_strategy_signature(self):
        return f"TopKThrLabeling_k{self.k}_thr{self.thr}"    
        
class LabelingContext:
    def __init__(self, labeling_strategy:LabelingStrategy):
        self.labeling_strategy = labeling_strategy

    def set_labeling_strategy(self, labeling_strategy):
        self.labeling_strategy = labeling_strategy
        
    def set_perc_dict(self, perc_dict):
        self.labeling_strategy.set_perc_dict(perc_dict)

    def process_annotation(self) -> list:
        return self.labeling_strategy.annotate()
