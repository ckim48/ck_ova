"""An unbiased finite-population weighted-gradient estimator on a toy model."""
from itertools import combinations
import torch
import torch.nn.functional as F
from pilot import weighted_gradient


class Data:
    x=torch.tensor([[1.,0.],[0.,1.],[1.,1.],[-1.,2.]])
    y=torch.tensor([0,1,0,1])
    def batch(self,ids,device):return self.x[ids].to(device),self.y[ids].to(device)


class Model:
    def __call__(self,x,theta):return x@theta.reshape(2,2)


def main():
    theta=torch.tensor([.3,-.2,.1,.4]);coefficients=torch.tensor([.1,.7,.2,.4]);dataset=Data();model=Model();device=torch.device('cpu')
    value=theta.detach().requires_grad_(True)
    target=torch.autograd.grad((F.cross_entropy(model(dataset.x,value),dataset.y,reduction='none')*coefficients).sum(),value)[0]
    estimates=[]
    for batch in combinations(range(4),2):
        ids=list(batch);weights=(4*coefficients[ids]).tolist()
        estimates.append(weighted_gradient(model,theta,dataset,{},ids,weights,device,1))
    torch.testing.assert_close(torch.stack(estimates).mean(dim=0),target,atol=1e-7,rtol=1e-6)
    unit=weighted_gradient(model,theta,dataset,{},[0,1,2,3],[1.,1.,1.,1.],device,2)
    value=theta.detach().requires_grad_(True)
    ordinary=torch.autograd.grad(F.cross_entropy(model(dataset.x,value),dataset.y),value)[0]
    torch.testing.assert_close(unit,ordinary,atol=1e-7,rtol=1e-6)
    print('PASS exact finite-population unbiasedness; microbatch and unit-weight consistency')


if __name__=='__main__':main()
