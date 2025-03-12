from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class MoE_LoRA(nn.Module):
    """.
    一个结合了MoE和LoRA的模块。
    """
    def __init__(self, r: int, dim: int, expert:int,input_size=(16,16), Dora=False, mode='train'):
        super(MoE_LoRA, self).__init__()
        assert r > 0
        # 需要有一个高秩的LoRA层，作为通用微调层
        self.Dora = Dora
        self.label_print = []
        self.num_experts = expert
        if self.Dora:
            self.general_layer = DoRALayer(dim, dim, r)
            self.specific_layers = nn.ModuleList([
                DoRALayer(dim, dim, r) for _ in range(expert)
            ])
        else:
            self.general_layer = LoRALayer(dim, dim, r)
            self.specific_layers = nn.ModuleList([
                LoRALayer(dim, dim, r) for _ in range(expert)
            ])
    def forward(self, x, label, weight):

        output = torch.zeros_like(x)
        # 原型路由
        for i in range(x.size(0)-1):  # x.size(0) 是 batch_size
            output[i] = self.specific_layers[label[i]](x[i]) + self.general_layer(x[i])
        return output


class MoE_Buffer(nn.Module):
    """.
    MoE模块的缓冲区,记录图像的embedding,用于模型的相似度对比
    """
    def __init__(self, num_experts, dim, mode, input_size=(16,16)):
        super(MoE_Buffer, self).__init__()
        self.num_experts = num_experts
        self.memory = nn.Parameter(torch.randn(num_experts,input_size[0]*input_size[1], dim))
        self.mode = mode
        self.count = {i: 0 for i in range(self.num_experts)}

    def init_prototypes(self, x):
        # 初始化原型
        self.memory = nn.Parameter(x)

    def cal_similarity(self, x):
        # 计算原型与输入的相似度
        x = x.unsqueeze(1).repeat(1,self.num_experts, 1, 1)
        batchsize = x.size(0)
        return F.cosine_similarity(x.view(batchsize, self.num_experts, -1), self.memory.view(self.num_experts, -1), dim=-1)
    
    def select_expert(self,x):
        # 通过原型与输入的相似度,选择最相似的原型
        sim = self.cal_similarity(x)

        return torch.argmax(sim, dim=-1)
    @torch.no_grad()
    def forward(self, x):
        batch, height, width, dim = x.size()
        x = x.view(batch, height*width, dim)
        buffer = torch.zeros_like(x).to(x.device)
        counts = {i: 0 for i in range(self.num_experts)}
        experts = self.select_expert(x)
        
        if self.mode == 'train':
            for i in range(batch):
                buffer[experts[i]] = (buffer[experts[i]] + x[i])
                counts[int(experts[i].item())] += 1
            for i in range(self.num_experts):
                if counts[i] == 0:
                    continue
                buffer[i] = buffer[i] / counts[i]
                self.memory[i] = (self.memory[i] + buffer[i]) / 2
            self.memory = nn.Parameter(self.memory.detach())
        elif self.mode == 'test':
            for expert in experts:
                expert_num = int(expert.item())
                self.count[expert_num] += 1
        return experts
        



class ExpertLoRA(nn.Module):
    def __init__(self, in_dim, out_dim, rank, q=True, k=False, v=True, o=True, alpha=1):
        super().__init__()
        self.q = q
        self.k = k
        self.v = v
        self.o = o
        if self.q:
            self.QLora = LoRALayer(in_dim, out_dim, rank, alpha)
        if self.k:
            self.KLora = LoRALayer(in_dim, out_dim, rank, alpha)
        if self.v:
            self.VLora = LoRALayer(in_dim, out_dim, rank, alpha)
        if self.o:
            self.OLora = LoRALayer(in_dim, out_dim, rank, alpha)
        

    def forward(self, x):
        q,k,v,o = None, None, None, None
        if self.q:
            q = self.QLora(x)
        if self.k:
            k = self.KLora(x)
        if self.v:
            v = self.VLora(x)
        if self.o:
            o = self.OLora(x)
        return q, k, v, o

class ExpertDoRA(nn.Module):
    def __init__(self, in_dim, out_dim, rank, q=True, k=False, v=True, o=True, alpha=1):
        super().__init__()
        self.q = q
        self.k = k
        self.v = v
        self.o = o
        # if self.q:
        #     self.QDora = DoRALayer(in_dim, out_dim, rank, alpha)
        # if self.k:
        #     self.KDora = DoRALayer(in_dim, out_dim, rank, alpha)
        # if self.v:
        #     self.VDora = DoRALayer(in_dim, out_dim, rank, alpha)
        # if self.o:
        #     self.ODora = DoRALayer(in_dim, out_dim, rank, alpha)
        

    def forward(self, x):
        q,k,v,o = None, None, None, None
        if self.q:
            q = self.QDora(x)
        if self.k:
            k = self.KDora(x)
        if self.v:
            v = self.VDora(x)
        if self.o:
            o = self.ODora(x)
        return q, k, v, o

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha=2):
        '''
        in_dim: the input dimension of the layer we want to modify using LoRA
        out_dim: the respective output dimension of that layer
        rank: a hyperparameter that controls the inner dimension of the matrices A and B
        alpha: a scaling hyperparameter applied to the output of the low-rank adaptation
        '''
        super().__init__()
        self.A = nn.Linear(in_dim, rank, bias=False)
        self.B = nn.Linear(rank, out_dim, bias=False)
        self.alpha = alpha
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        x = self.alpha * self.B(self.A(x))
        return x
    
class DoRALayer(nn.Module):
    """
    一个结合了Linear层和DoRA层的模块。
    该模块通过使用DoRA技术来增强线性层的表示能力。

    参数:
    - linear: 原始的线性层(nn.Linear)实例
    - rank: DoRA层的秩
    - alpha: DoRA层中的缩放参数
    """

    def __init__(self, linear, rank, alpha=1):
        super().__init__()
        self.linear = linear  # 原始的线性层
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )  # 初始化LoRA层
        self.m = nn.Parameter(
            self.linear.weight.norm(p=2, dim=0, keepdim=True)
        )  # 初始化一个参数，用于调整新权重的模长

    # Code loosely inspired by    
    # https://github.com/catid/dora/blob/main/dora.py
    def forward(self, x):
        """
        前向传播过程。
        参数:
        - x: 输入特征向量

        返回:
        - 经过更新后的线性变换后的结果
        """
        lora = self.lora.A(self.lora.B) # 计算LoRA的低秩近似
        numerator = self.linear.weight + self.lora.alpha*lora.T
        denominator = numerator.norm(p=2, dim=0, keepdim=True)
        directional_component = numerator / denominator # normalize，计算方向组件
        new_weight = self.m * directional_component # 更新权重
        return F.linear(x, new_weight, self.linear.bias) # 应用新的权重进行线性变换
    

if __name__ == '__main__':
    buffer = MoE_Buffer(16, 768,mode='train')
    buffer.init_prototypes(torch.rand(16, 256, 768))
    input = torch.rand(40, 16, 16, 768)
    expert = buffer(input)
    print(expert)