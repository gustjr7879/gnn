Sampling과 Aggregation을 학습시키는 GraphSAGE 모델을 dgl과 torch를 사용하여서 구현해보았다.

좀 뒤늦게 안 사실이지만 label을 torch.tensor로 보내서 처리하게 되면 loss 계산 시 문제가 발생할 수 있다는 것도 알게 되었음.

