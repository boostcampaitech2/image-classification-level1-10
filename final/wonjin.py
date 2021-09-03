# FocalLoss를 사용하는데 여기에 Label Smoothing을 적용하기 위해 아래와 같이
# Loss를 작성했는데 이런식으로 작성하는게 맞는지 궁금합니다!!.
# 참고로 저는 Mask, Gender, Age별로 나눠서 모델을 만들었기 때문에
# 아래와 같이 cls를 정의해서 output의 dimension을 결정했습니다.
class CustomFocalLoss(nn.Module):
    """
    CustomFocalLoss with Label smoothing
    """

    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.smoothing = 0.01

    def forward(self, input_tensor, target_tensor, kind):
        if kind == "mask" or kind == "age":
            cls = 3
        elif kind == "gender":
            cls = 2
        else:
            raise ValueError("kind does not exist")
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        with torch.no_grad():
            true_dist = torch.zeros_like(input_tensor)
            true_dist.fill_(self.smoothing)
            true_dist.scatter_(
                1, target_tensor.data.unsqueeze(1), 1-(self.smoothing*(cls-1)))
        ratio = ((1 - prob) ** self.gamma) * log_prob
        return torch.mean(torch.sum(-ratio*true_dist, dim=-1))
        # return F.nll_loss(
        #     ((1 - prob) ** self.gamma) * log_prob,
        #     target_tensor,
        #     weight = self.weight,
        #     reduction = self.reduction
        # )