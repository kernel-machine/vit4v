class ValidationMetrics:
    def __init__(self):
        self.predictions = []
        self.labels = []

    def add_prediction(self, prediction:bool, label:bool):
        self.predictions.append(prediction)
        self.labels.append(label)

    def get_metrics(self) -> tuple[int,int,int,int]:
        assert len(self.predictions)==len(self.labels)
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(len(self.predictions)):
            if self.predictions[i] and self.labels[i]:
                tp+=1
            elif self.predictions[i] and not self.labels[i]:
                fp+=1
            elif not self.predictions[i] and not self.labels[i]:
                tn+=1
            elif not self.predictions[i] and self.labels[i]:
                fn+=1
        return tp,fp,tn,fn
    
    def get_precision(self)->float:
        tp, fp, _, _ = self.get_metrics()
        if tp+fp == 0:
            return 0
        return tp/(tp+fp)
    
    def get_recall(self)->float:
        _, _, tn, fn = self.get_metrics()
        if tn + fn == 0:
            return 0
        return tn/(tn+fn)
    
    def get_f1(self)->float:
        precision = self.get_precision()
        recall = self.get_recall()
        if precision + recall == 0:
            return 0
        return 2*((precision*recall)/(precision+recall))
    
    def get_accuracy(self)->float:
        tp, fp, tn, fn = self.get_metrics()
        return (tp+tn)/(tp+fp+tn+fn)
    
    def __str__(self):
        tp, fp, tn, fn = self.get_metrics()
        return f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn} | Acc: {self.get_accuracy()} | F1: {self.get_f1()}"