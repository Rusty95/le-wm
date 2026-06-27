import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """
    AdaLN-zero 调制函数 - 用于自适应层归一化中的仿射变换
    
    作用: 通过shift和scale参数对输入进行自适应的仿射变换，实现条件生成
    
    公式: y = x * (1 + scale) + shift
    
    参数:
        x (Tensor): 输入特征，形状 (B, T, D) 或 (B, D)
        shift (Tensor): 偏移量，shape与x一致
        scale (Tensor): 缩放因子，shape与x一致
    
    返回:
        Tensor: 调制后的特征
    
    使用案例:
        >>> x = torch.randn(2, 10, 512)           # (batch, time, dim)
        >>> shift = torch.randn(2, 10, 512)       # 偏移参数
        >>> scale = torch.randn(2, 10, 512)       # 缩放参数
        >>> y = modulate(x, shift, scale)
        >>> # y = x * (1 + scale) + shift
        >>> # 例如: x[0,0,:] = [1.0, 2.0] 
        >>> #       shift[0,0,:] = [0.1, 0.2]
        >>> #       scale[0,0,:] = [0.5, -0.3]
        >>> #       结果: y[0,0,:] = [1.0*(1+0.5)+0.1, 2.0*(1-0.3)+0.2]
        >>> #                      = [1.6, 2.2]
    """
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """
    素描等向高斯正则化器 (Sketch Isotropic Gaussian Regularizer) - 单GPU优化
    
    作用: 通过约束隐空间嵌入遵循标准高斯分布来防止表示坍缩
    
    核心思想:
        1. 使用高斯检验来验证嵌入是否满足标准高斯分布 N(0, I)
        2. 采用Sketch方法（随机投影）降低计算复杂度
        3. 不需要跨GPU同步，单GPU友好
    
    参数:
        knots (int): 高斯检验中的节点数，默认17
        num_proj (int): 随机投影数，默认1024
    
    缓冲区:
        t: 高斯节点位置 (0到3的均匀分布)
        phi: 高斯窗口 exp(-t²/2)
        weights: 积分权重 (梯形积分)
    """

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        # 高斯检验的节点：从0到3均匀分布（覆盖标准高斯的主要部分）
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        # 梯形积分规则的权重：两端为dt，中间为2*dt
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        # 高斯窗口函数：exp(-t²/2)
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        计算嵌入的高斯正则化损失
        
        参数:
            proj (Tensor): 嵌入张量，形状 (T, B, D)
                          其中 T=时间步, B=批次, D=特征维度
        
        返回:
            Tensor: 标量损失值
        
        处理流程:
            1. 生成随机投影矩阵 A: (D, num_proj)
            2. 投影嵌入: z_proj = proj @ A，形状 (T, B, num_proj)
            3. 计算高斯统计: 检查cos/sin特征是否匹配高斯分布
            4. 返回平均损失
        
        使用案例:
            >>> sigreg = SIGReg(knots=17, num_proj=1024)
            >>> emb = torch.randn(10, 8, 512)  # (T=10, B=8, D=512)
            >>> loss = sigreg(emb.transpose(0, 1))  # 需要 (T, B, D) 格式
            >>> loss.backward()
            >>> # 损失值在0.0-2.0之间，值越小表示嵌入越接近高斯分布
            >>> print(loss.item())  # 例如: 0.1234
        
        详细计算步骤:
            假设 proj = (T=2, B=3, D=4)，num_proj=2
            
            1. 生成投影矩阵 A: (4, 2)
               A = [[0.5, -0.2],
                    [0.1,  0.8],
                    [0.3,  0.1],
                    [-0.4, 0.6]]
            
            2. 投影: x_t = (proj @ A) * t.unsqueeze(-1)
               得到 x_t: (T=2, B=3, num_proj=2, knots=17)
            
            3. 计算特征:
               cos特征: cos(x_t).mean(dim=-3) → (B, num_proj, knots)
               sin特征: sin(x_t).mean(dim=-3) → (B, num_proj, knots)
            
            4. 与高斯窗口比较:
               err = (cos_mean - phi)² + sin_mean²
            
            5. 加权求和和平均
        """
        # 生成随机投影矩阵：用于将高维嵌入投影到低维
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        # L2正规化每个投影方向（确保投影是单位向量）
        A = A.div_(A.norm(p=2, dim=0))
        
        # 投影嵌入到随机基上，并乘以高斯节点位置
        # x_t: (T, B, num_proj, knots)
        x_t = (proj @ A).unsqueeze(-1) * self.t
        
        # 计算高斯检验统计量
        # 根据中心极限定理，标准高斯在cos/sin投影下的期望值
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        
        # 梯形积分规则的加权求和
        statistic = (err @ self.weights) * proj.size(-2)
        
        # 返回所有投影和时间步的平均损失
        return statistic.mean()
    
class FeedForward(nn.Module):
    """
    前馈网络 - Transformer中的全连接层序列
    
    作用: 在Transformer中进行非线性特征变换
    
    结构: LayerNorm → Linear(dim→hidden_dim) → GELU → Dropout → 
          Linear(hidden_dim→dim) → Dropout
    
    参数:
        dim (int): 输入输出维度
        hidden_dim (int): 隐层维度（通常为dim的2-4倍）
        dropout (float): Dropout概率
    
    使用案例:
        >>> ffn = FeedForward(dim=512, hidden_dim=2048, dropout=0.1)
        >>> x = torch.randn(2, 10, 512)  # (batch, seq_len, dim)
        >>> y = ffn(x)
        >>> print(y.shape)  # torch.Size([2, 10, 512])
        
        >>> # 训练时有dropout
        >>> ffn.train()
        >>> y_train = ffn(x)
        
        >>> # 评估时没有dropout
        >>> ffn.eval()
        >>> y_eval = ffn(x)
    """

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            # 第1步：层归一化，稳定训练
            nn.LayerNorm(dim),
            # 第2步：线性层，维度扩展 dim → hidden_dim
            nn.Linear(dim, hidden_dim),
            # 第3步：GELU激活函数，光滑的非线性变换
            nn.GELU(),
            # 第4步：Dropout，防止过拟合
            nn.Dropout(dropout),
            # 第5步：线性层，维度压缩回原始 hidden_dim → dim
            nn.Linear(hidden_dim, dim),
            # 第6步：Dropout，防止过拟合
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        前向传播
        
        参数:
            x (Tensor): 输入，形状 (B, T, dim) 或其他
        
        返回:
            Tensor: 输出，形状与输入相同
        
        使用案例:
            >>> x = torch.randn(4, 20, 512)
            >>> ffn = FeedForward(512, 2048, dropout=0.1)
            >>> out = ffn(x)
            >>> assert out.shape == x.shape
        """
        return self.net(x)


class Attention(nn.Module):
    """
    缩放点积注意力 - 支持因果掩码的多头注意力机制
    
    作用: 实现自注意力机制，使序列中的每个位置可以关注其他位置的信息
    
    数学原理:
        Attention(Q, K, V) = softmax(Q·K^T / √(d_k))·V
        
        其中Q、K、V分别是query、key、value投影
    
    参数:
        dim (int): 输入维度
        heads (int): 注意力头数，默认8
        dim_head (int): 每个头的维度，默认64
        dropout (float): Dropout概率
    
    使用案例:
        >>> attn = Attention(dim=512, heads=8, dim_head=64, dropout=0.1)
        >>> x = torch.randn(2, 10, 512)  # (batch, seq_len, dim)
        
        >>> # 自注意力（默认因果掩码）
        >>> y = attn(x, causal=True)
        >>> print(y.shape)  # torch.Size([2, 10, 512])
        
        >>> # 双向注意力
        >>> y = attn(x, causal=False)
        
        >>> # 计算维度
        >>> inner_dim = 64 * 8 = 512
        >>> # Q、K、V投影后都是 (batch, heads, seq_len, dim_head)
        >>> # 注意力权重: (batch, heads, seq_len, seq_len)
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        # 多头的总维度
        inner_dim = dim_head * heads
        # 判断是否需要输出投影（当输出维度与输入不同时）
        project_out = not (heads == 1 and dim_head == dim)
        
        self.heads = heads
        # 注意力温度系数：较小的值使softmax更尖锐
        self.scale = dim_head**-0.5
        self.dropout = dropout
        
        # 层归一化，稳定注意力权重分布
        self.norm = nn.LayerNorm(dim)
        # Softmax用于计算注意力权重
        self.attend = nn.Softmax(dim=-1)
        
        # 线性层：投影输入到Q、K、V (3*inner_dim维)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        # 输出投影：将多头的输出拼接后投影回dim维
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        前向传播 - 计算注意力输出
        
        参数:
            x (Tensor): 输入，形状 (B, T, D)
                       B=批次, T=序列长度, D=维度
            causal (bool): 是否使用因果掩码（默认True用于自回归）
                          因果掩码使token只能关注当前及之前的位置
        
        返回:
            Tensor: 注意力输出，形状 (B, T, D)
        
        详细流程:
            1. 层归一化: (B, T, D) → (B, T, D)
            2. 投影到Q、K、V: (B, T, 3*inner_dim)
            3. 分割为Q、K、V: 各 (B, T, inner_dim)
            4. 重排为多头格式: (B, heads, T, dim_head)
            5. 计算注意力: (B, heads, T, T)
            6. 应用Dropout（训练时）
            7. 与value相乘: (B, heads, T, dim_head)
            8. 重排回序列格式: (B, T, inner_dim)
            9. 输出投影: (B, T, D)
        
        使用案例:
            >>> attn = Attention(dim=512, heads=8, dim_head=64, dropout=0.1)
            >>> x = torch.randn(2, 10, 512)
            
            >>> # 自回归模式（因果掩码）- 用于预测任务
            >>> y_causal = attn(x, causal=True)
            >>> # token在位置t只能看到位置0..t的信息
            
            >>> # 双向模式 - 用于编码器
            >>> y_bidir = attn(x, causal=False)
            >>> # token在位置t可以看到所有位置的信息
            
            >>> print(y_causal.shape)  # torch.Size([2, 10, 512])
        """
        # 层归一化
        x = self.norm(x)
        # 训练时使用dropout，评估时不使用
        drop = self.dropout if self.training else 0.0
        
        # 投影到Q、K、V并分割
        # qkv: 3个 (B, T, inner_dim)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        
        # 重排维度以支持多头
        # 从 (B, T, inner_dim) → (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        
        # 使用PyTorch的高效SDPA实现注意力
        # 自动应用因果掩码（如果is_causal=True）
        out = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=drop,        # Dropout在attention权重上
            is_causal=causal       # 是否使用因果掩码
        )
        
        # 重排回序列格式
        # 从 (B, heads, T, dim_head) → (B, T, inner_dim)
        out = rearrange(out, "b h t d -> b t (h d)")
        
        # 输出投影
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """
    条件Transformer块 - 支持AdaLN-zero调制的Transformer层
    
    作用: 在Transformer块的基础上添加条件调制功能
    
    核心创新 (AdaLN-zero):
        - 使用从条件向量c导出的参数来调制每个子层的归一化参数
        - 初始化为0，使得在训练初期不影响原始表示
        - 实现了残差的零初始化技巧
    
    结构:
        输入 → [AdaLN(attn) + 门控残差] → [AdaLN(mlp) + 门控残差] → 输出
    
    参数:
        dim (int): 特征维度
        heads (int): 注意力头数
        dim_head (int): 每个头的维度
        mlp_dim (int): FFN隐层维度
        dropout (float): Dropout概率
    
    使用案例:
        >>> cond_block = ConditionalBlock(dim=512, heads=8, dim_head=64, 
        ...                               mlp_dim=2048, dropout=0.1)
        >>> x = torch.randn(2, 10, 512)  # 输入特征
        >>> c = torch.randn(2, 512)      # 条件向量
        >>> y = cond_block(x, c)
        >>> print(y.shape)  # torch.Size([2, 10, 512])
    """

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        # 注意力层
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        # FFN层
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        
        # 不使用elementwise_affine的LayerNorm（因为affine由AdaLN提供）
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        
        # AdaLN调制网络：从条件向量c生成6个调制参数
        # shift_msa, scale_msa, gate_msa (用于注意力层)
        # shift_mlp, scale_mlp, gate_mlp (用于MLP层)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),                      # 激活函数
            nn.Linear(dim, 6 * dim, bias=True)  # 投影到6*dim维
        )

        # 零初始化：关键技巧！确保训练初期不改变原始表示
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        """
        前向传播 - 带条件调制的Transformer块
        
        参数:
            x (Tensor): 输入特征，形状 (B, T, dim)
            c (Tensor): 条件向量，形状 (B, dim)
                       通常由上一层的输出（time token）提供
        
        返回:
            Tensor: 输出特征，形状 (B, T, dim)
        
        处理流程:
            1. 从条件向量c生成6个调制参数
            2. 对注意力层应用AdaLN调制和门控残差
            3. 对MLP层应用AdaLN调制和门控残差
            4. 返回调制后的特征
        
        使用案例:
            >>> # 模拟一个条件生成场景
            >>> batch_size, seq_len, dim = 4, 20, 512
            >>> block = ConditionalBlock(dim=512, heads=8, dim_head=64, 
            ...                         mlp_dim=2048, dropout=0.1)
            >>> 
            >>> # 输入序列
            >>> x = torch.randn(batch_size, seq_len, dim)
            >>> # 条件向量（可能来自其他模态或时间步编码）
            >>> c = torch.randn(batch_size, dim)
            >>> 
            >>> # 前向传播
            >>> y = block(x, c)
            >>> print(y.shape)  # torch.Size([4, 20, 512])
            >>> 
            >>> # 验证残差连接
            >>> assert not torch.allclose(y, x)  # 输出应该不同
        
        详细计算:
            假设 x: (B=2, T=5, dim=512), c: (B=2, dim=512)
            
            1. 生成调制参数:
               shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp 
               = adaLN_modulation(c).chunk(6, dim=-1)
               
               每个形状: (B=2, dim=512)
            
            2. 注意力层:
               x_norm1 = norm1(x)  # (B, T, dim)
               x_mod = modulate(x_norm1, shift_msa, scale_msa)  # AdaLN调制
               x_attn = attn(x_mod)  # 多头自注意力
               x = x + gate_msa * x_attn  # 门控残差 (gate_msa 从c导出)
               
               门的作用：当gate_msa≈0时，残差很小；当gate_msa≈1时，残差全部加入
            
            3. MLP层:
               x_norm2 = norm2(x)  # (B, T, dim)
               x_mod = modulate(x_norm2, shift_mlp, scale_mlp)  # AdaLN调制
               x_mlp = mlp(x_mod)  # 前馈网络
               x = x + gate_mlp * x_mlp  # 门控残差
            
            4. 返回 x: (B=2, T=5, dim=512)
        """
        # 从条件向量生成调制参数
        # chunk(6, dim=-1) 沿最后一个维度分割为6个张量
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        
        # 注意力层（带AdaLN调制和门控残差）
        # 步骤1：对输入进行归一化
        x_attn_input = self.norm1(x)
        # 步骤2：应用条件调制（shift和scale）
        x_attn_modulated = modulate(x_attn_input, shift_msa, scale_msa)
        # 步骤3：计算注意力
        x_attn = self.attn(x_attn_modulated)
        # 步骤4：门控残差连接
        # gate_msa 的值由条件c决定，实现自适应的信息流量控制
        x = x + gate_msa * x_attn
        
        # MLP层（带AdaLN调制和门控残差）
        # 步骤1：对输入进行归一化
        x_mlp_input = self.norm2(x)
        # 步骤2：应用条件调制
        x_mlp_modulated = modulate(x_mlp_input, shift_mlp, scale_mlp)
        # 步骤3：计算MLP
        x_mlp = self.mlp(x_mlp_modulated)
        # 步骤4：门控残差连接
        x = x + gate_mlp * x_mlp
        
        return x


class Block(nn.Module):
    """
    标准Transformer块 - 无条件调制的基础Transformer层
    
    作用: 实现标准的Pre-LN Transformer块
    
    结构:
        输入 → [LayerNorm + Attention + 残差] → 
               [LayerNorm + MLP + 残差] → 输出
    
    参数:
        dim (int): 特征维度
        heads (int): 注意力头数
        dim_head (int): 每个头的维度
        mlp_dim (int): FFN隐层维度
        dropout (float): Dropout概率
    
    使用案例:
        >>> block = Block(dim=512, heads=8, dim_head=64, 
        ...              mlp_dim=2048, dropout=0.1)
        >>> x = torch.randn(2, 10, 512)
        >>> y = block(x)
        >>> print(y.shape)  # torch.Size([2, 10, 512])
    """

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        # 多头自注意力层
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        # 前馈网络层
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        # Pre-LN架构：在子层之前进行归一化
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        """
        前向传播 - 标准Transformer块
        
        参数:
            x (Tensor): 输入，形状 (B, T, dim)
        
        返回:
            Tensor: 输出，形状 (B, T, dim)
        
        处理流程:
            1. 注意力层：x = x + Attention(LayerNorm(x))
            2. MLP层：x = x + MLP(LayerNorm(x))
            3. 返回处理后的x
        
        使用案例:
            >>> block = Block(dim=512, heads=8, dim_head=64, mlp_dim=2048)
            >>> x = torch.randn(8, 20, 512)  # (batch=8, seq_len=20, dim=512)
            >>> 
            >>> # 前向传播
            >>> y = block(x)
            >>> assert y.shape == x.shape
            >>> 
            >>> # 验证残差连接
            >>> with torch.no_grad():
            ...     y2 = block(x)
            ...     # 由于有dropout，两次前向可能略有不同（训练模式）
            >>> 
            >>> block.eval()  # 评估模式
            >>> y_eval = block(x)
            >>> y_eval2 = block(x)
            >>> assert torch.allclose(y_eval, y_eval2)  # 确定性输出
        
        对比ConditionalBlock:
            Block:
            - 无条件输入
            - 固定的层参数
            - 用于标准编码器
            - 更快更简单
            
            ConditionalBlock:
            - 需要条件向量c
            - 参数由条件导出
            - 用于条件生成
            - 更灵活更复杂
        """
        # 自注意力分支：x = x + Attention(LayerNorm(x))
        # Pre-LN架构确保数值稳定性
        x = x + self.attn(self.norm1(x))
        
        # MLP分支：x = x + MLP(LayerNorm(x))
        x = x + self.mlp(self.norm2(x))
        
        return x


class Transformer(nn.Module):
    """
    Transformer堆叠 - 支持Block和ConditionalBlock的灵活堆叠
    
    作用: 将多个Transformer块堆叠在一起，形成深层网络
    
    特点:
        - 支持动态block类选择（Block或ConditionalBlock）
        - 自动处理维度不匹配（投影层）
        - 灵活的输入输出维度变换
    
    参数:
        input_dim (int): 输入维度
        hidden_dim (int): 隐层维度
        output_dim (int): 输出维度
        depth (int): Transformer块的层数
        heads (int): 注意力头数
        dim_head (int): 每个头的维度
        mlp_dim (int): FFN隐层维度
        dropout (float): Dropout概率
        block_class: 块类型（Block或ConditionalBlock）
    
    使用案例:
        >>> # 标准Transformer编码器
        >>> encoder = Transformer(
        ...     input_dim=768,      # Vision Transformer输出
        ...     hidden_dim=512,
        ...     output_dim=512,
        ...     depth=6,
        ...     heads=8,
        ...     dim_head=64,
        ...     mlp_dim=2048,
        ...     dropout=0.1,
        ...     block_class=Block
        ... )
        >>> x = torch.randn(2, 10, 768)
        >>> y = encoder(x)
        >>> print(y.shape)  # torch.Size([2, 10, 512])
        
        >>> # 条件Transformer（用于动作预测）
        >>> predictor = Transformer(
        ...     input_dim=512,
        ...     hidden_dim=512,
        ...     output_dim=512,
        ...     depth=4,
        ...     heads=8,
        ...     dim_head=64,
        ...     mlp_dim=2048,
        ...     dropout=0.1,
        ...     block_class=ConditionalBlock
        ... )
        >>> x = torch.randn(2, 10, 512)       # 特征序列
        >>> c = torch.randn(2, 512)           # 动作条件
        >>> y = predictor(x, c)
        >>> print(y.shape)  # torch.Size([2, 10, 512])
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        # 输出的最终归一化
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        # 输入投影：如果维度不匹配则进行投影，否则使用恒等映射
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        # 条件投影：用于ConditionalBlock的条件向量投影
        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        # 输出投影：如果维度不匹配则进行投影，否则使用恒等映射
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        # 堆叠多个Transformer块
        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):
        """
        前向传播 - Transformer堆叠
        
        参数:
            x (Tensor): 输入特征，形状 (B, T, input_dim)
            c (Tensor): 可选条件向量，形状 (B, cond_dim)
                       仅在使用ConditionalBlock时需要
        
        返回:
            Tensor: 输出，形状 (B, T, output_dim)
        
        处理流程:
            1. 输入投影：input_dim → hidden_dim
            2. 条件投影（如果需要）：cond_dim → hidden_dim
            3. 依次通过depth个Transformer块
            4. 最终归一化
            5. 输出投影：hidden_dim → output_dim
        
        使用案例:
            >>> # Case 1: 标准编码器堆叠
            >>> transformer = Transformer(
            ...     input_dim=768, hidden_dim=512, output_dim=256,
            ...     depth=6, heads=8, dim_head=64, mlp_dim=2048,
            ...     block_class=Block
            ... )
            >>> x = torch.randn(4, 20, 768)
            >>> y = transformer(x)  # 只需要x
            >>> print(y.shape)  # torch.Size([4, 20, 256])
            
            >>> # Case 2: 条件预测器堆叠
            >>> predictor = Transformer(
            ...     input_dim=512, hidden_dim=512, output_dim=512,
            ...     depth=4, heads=8, dim_head=64, mlp_dim=2048,
            ...     block_class=ConditionalBlock
            ... )
            >>> x = torch.randn(4, 10, 512)  # 嵌入序列
            >>> c = torch.randn(4, 512)      # 动作特征
            >>> y = predictor(x, c)  # 需要x和c
            >>> print(y.shape)  # torch.Size([4, 10, 512])
            
            >>> # Case 3: 维度变换
            >>> transformer = Transformer(
            ...     input_dim=1024, hidden_dim=768, output_dim=256,
            ...     depth=12, heads=12, dim_head=64, mlp_dim=3072,
            ...     block_class=Block
            ... )
            >>> x = torch.randn(2, 16, 1024)
            >>> y = transformer(x)
            >>> print(y.shape)  # torch.Size([2, 16, 256])
            
            详细流程:
            --------
            输入 (B=2, T=10, input_dim=768)
                ↓
            input_proj (768→512)
                ↓ (B, T, 512)
            Block 1 × depth
                ↓ (B, T, 512)
            LayerNorm
                ↓ (B, T, 512)
            output_proj (512→256)
                ↓ (B, T, 256)
            输出
        """
        # 第1步：输入投影
        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        # 第2步：条件投影（如果存在）
        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        # 第3步：通过Transformer块堆叠
        for block in self.layers:
            # 判断是Block还是ConditionalBlock
            # Block: forward(x)
            # ConditionalBlock: forward(x, c)
            x = block(x) if isinstance(block, Block) else block(x, c)
        
        # 第4步：最终归一化
        x = self.norm(x)

        # 第5步：输出投影
        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        
        return x

class Embedder(nn.Module):
    """
    1D嵌入器 - 用于时间序列特征的编码
    
    作用: 将动作或其他1D序列特征编码为高维嵌入向量
    
    结构:
        输入 → Conv1d平滑 → MLP编码 → 输出嵌入
    
    参数:
        input_dim (int): 输入特征维度（如动作维度）
        smoothed_dim (int): Conv1d输出维度
        emb_dim (int): 最终嵌入维度
        mlp_scale (int): MLP隐层相对大小倍数
    
    使用案例:
        >>> embedder = Embedder(input_dim=10, smoothed_dim=16, 
        ...                     emb_dim=128, mlp_scale=4)
        >>> # 编码动作序列
        >>> action = torch.randn(2, 5, 10)  # (batch, time, action_dim)
        >>> emb = embedder(action)
        >>> print(emb.shape)  # torch.Size([2, 5, 128])
        
        >>> # 不同的输入大小
        >>> action_long = torch.randn(4, 20, 10)
        >>> emb_long = embedder(action_long)
        >>> print(emb_long.shape)  # torch.Size([4, 20, 128])
    """
    
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        
        # Conv1d平滑层：用于时间维度上的局部聚合
        # kernel_size=1表示不跨越时间步，只在特征维度上投影
        self.patch_embed = nn.Conv1d(
            input_dim,        # 输入通道
            smoothed_dim,     # 输出通道
            kernel_size=1,    # 1x1卷积
            stride=1
        )
        
        # MLP编码：将平滑后的特征映射到最终嵌入空间
        self.embed = nn.Sequential(
            # 扩展投影
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            # 非线性激活
            nn.SiLU(),
            # 压缩投影
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        前向传播 - 编码1D序列特征
        
        参数:
            x (Tensor): 输入序列，形状 (B, T, input_dim)
                       B=批次, T=时间步, input_dim=特征维度
        
        返回:
            Tensor: 嵌入向量，形状 (B, T, emb_dim)
        
        处理流程:
            1. 转换为浮点数
            2. 置换维度以适应Conv1d: (B, T, D) → (B, D, T)
            3. Conv1d平滑: (B, D, T) → (B, smoothed_dim, T)
            4. 置换回来: (B, smoothed_dim, T) → (B, T, smoothed_dim)
            5. MLP编码: (B, T, smoothed_dim) → (B, T, emb_dim)
        
        使用案例:
            >>> embedder = Embedder(input_dim=6, smoothed_dim=16, 
            ...                     emb_dim=128, mlp_scale=4)
            >>> 
            >>> # Case 1: 机器人动作编码
            >>> # 假设6维度动作 (6-DOF机械臂)
            >>> action = torch.randn(2, 10, 6)  # (batch=2, time=10, action_dim=6)
            >>> action_emb = embedder(action)
            >>> print(action_emb.shape)  # torch.Size([2, 10, 128])
            
            >>> # Case 2: 长序列处理
            >>> long_action = torch.randn(4, 50, 6)
            >>> long_emb = embedder(long_action)
            >>> print(long_emb.shape)  # torch.Size([4, 50, 128])
            
            >>> # Case 3: 验证数值正常性
            >>> print(f"输入范围: [{action.min():.3f}, {action.max():.3f}]")
            >>> print(f"嵌入范围: [{action_emb.min():.3f}, {action_emb.max():.3f}]")
            
            详细流程示例:
            输入: x = (B=2, T=5, D=6)
                例如: [[[a0, a1, ..., a5],
                        [b0, b1, ..., b5],
                        ...]]
                ↓
            permute(0, 2, 1): (2, 6, 5)
                例如: [[[a0, b0, ...],
                        [a1, b1, ...],
                        ...,
                        [a5, b5, ...]]]
                ↓
            Conv1d (6→16): (2, 16, 5)
                进行1x1卷积，将6维度映射到16维度
                ↓
            permute(0, 2, 1): (2, 5, 16)
                恢复序列格式
                ↓
            MLP (16→512→128): (2, 5, 128)
                两层全连接，中间层512维
                ↓
            输出: (2, 5, 128)
        """
        # 转换为浮点张量
        x = x.float()
        
        # 置换维度以适应Conv1d
        # Conv1d期望输入形状: (batch, channels, length)
        # 从 (B, T, D) → (B, D, T)
        x = x.permute(0, 2, 1)
        
        # 1x1卷积进行投影和平滑
        # 从 (B, D, T) → (B, smoothed_dim, T)
        x = self.patch_embed(x)
        
        # 置换回序列格式
        # 从 (B, smoothed_dim, T) → (B, T, smoothed_dim)
        x = x.permute(0, 2, 1)
        
        # MLP编码
        # 从 (B, T, smoothed_dim) → (B, T, emb_dim)
        x = self.embed(x)
        
        return x


class MLP(nn.Module):
    """
    简单多层感知机 - 支持可选的归一化和激活函数
    
    作用: 用于通用的特征变换和映射
    
    结构: Linear → Norm → Activation → Linear
    
    参数:
        input_dim (int): 输入维度
        hidden_dim (int): 隐层维度
        output_dim (int): 输出维度（可选，默认=input_dim）
        norm_fn: 归一化函数（可选，默认LayerNorm）
        act_fn: 激活函数（可选，默认GELU）
    
    使用案例:
        >>> mlp = MLP(input_dim=512, hidden_dim=1024, output_dim=256)
        >>> x = torch.randn(16, 512)
        >>> y = mlp(x)
        >>> print(y.shape)  # torch.Size([16, 256])
        
        >>> # 无输出映射（维度保持）
        >>> mlp2 = MLP(input_dim=512, hidden_dim=1024)
        >>> y2 = mlp2(x)
        >>> print(y2.shape)  # torch.Size([16, 512])
        
        >>> # 自定义归一化和激活
        >>> mlp3 = MLP(input_dim=512, hidden_dim=1024, 
        ...           norm_fn=nn.BatchNorm1d, act_fn=nn.ReLU)
        >>> y3 = mlp3(x)
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        
        # 创建归一化层（如果指定）
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        
        # 构建MLP序列
        self.net = nn.Sequential(
            # 第1层：输入映射到隐层
            nn.Linear(input_dim, hidden_dim),
            # 归一化
            norm_fn,
            # 激活函数
            act_fn(),
            # 第2层：隐层映射到输出
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        前向传播 - MLP处理
        
        参数:
            x (Tensor): 输入，形状 (B*T, input_dim) 或 (B, input_dim)
                       *表示任意维度组合
        
        返回:
            Tensor: 输出，形状 (B*T, output_dim) 或 (B, output_dim)
        
        使用案例:
            >>> mlp = MLP(input_dim=512, hidden_dim=2048, output_dim=256)
            
            >>> # Case 1: 2D输入 (batch, dim)
            >>> x_2d = torch.randn(32, 512)
            >>> y_2d = mlp(x_2d)
            >>> print(y_2d.shape)  # torch.Size([32, 256])
            
            >>> # Case 2: 3D输入 (batch, time, dim)
            >>> x_3d = torch.randn(32, 10, 512)
            >>> # 需要reshape为2D
            >>> x_2d_reshaped = x_3d.reshape(-1, 512)  # (320, 512)
            >>> y_2d = mlp(x_2d_reshaped)
            >>> y_3d = y_2d.reshape(32, 10, 256)  # reshape回3D
            >>> print(y_3d.shape)  # torch.Size([32, 10, 256])
            
            >>> # Case 3: 不同的配置
            >>> # 配置1：输出维度不变
            >>> mlp_identity = MLP(512, 2048, output_dim=512)
            >>> y = mlp_identity(x_2d)
            >>> assert y.shape == x_2d.shape
            
            >>> # 配置2：无归一化
            >>> mlp_no_norm = MLP(512, 2048, output_dim=256, norm_fn=None)
            >>> y = mlp_no_norm(x_2d)
            >>> assert y.shape[1] == 256
            
            >>> # 配置3：ReLU激活而非GELU
            >>> mlp_relu = MLP(512, 2048, output_dim=256, act_fn=nn.ReLU)
            >>> y = mlp_relu(x_2d)
            
            处理示例:
            输入: x = (B*T=32, input_dim=512)
            例如: x[0] = [0.1, -0.2, 0.3, ..., -0.5]
                ↓
            Linear(512→2048): (32, 2048)
                ↓
            LayerNorm: (32, 2048)
                规范化到均值0方差1
                ↓
            GELU: (32, 2048)
                平滑的非线性变换
                ↓
            Linear(2048→256): (32, 256)
                ↓
            输出: (32, 256)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """
    自回归预测器 - 用于下一步嵌入预测的Transformer
    
    作用: 基于上文嵌入和动作预测下一时刻的嵌入向量
    
    特点:
        - 包含可学习的位置编码
        - 使用ConditionalBlock支持动作条件
        - 输出与输入维度相同
    
    参数:
        num_frames (int): 最大序列长度
        depth (int): Transformer层数
        heads (int): 注意力头数
        mlp_dim (int): FFN隐层维度
        input_dim (int): 输入嵌入维度
        hidden_dim (int): 隐层维度
        output_dim (int): 输出维度（可选）
        dim_head (int): 每个头的维度
        dropout (float): Dropout概率
        emb_dropout (float): 嵌入Dropout概率
    
    使用案例:
        >>> predictor = ARPredictor(
        ...     num_frames=10,
        ...     depth=4,
        ...     heads=8,
        ...     mlp_dim=2048,
        ...     input_dim=512,
        ...     hidden_dim=512,
        ...     output_dim=512,
        ...     dim_head=64,
        ...     dropout=0.1,
        ...     emb_dropout=0.1
        ... )
        >>> 
        >>> # 上文嵌入
        >>> x = torch.randn(2, 5, 512)   # (batch, context_len, dim)
        >>> # 动作编码
        >>> c = torch.randn(2, 512)      # (batch, action_emb_dim)
        >>> 
        >>> # 预测下一步嵌入
        >>> pred = predictor(x, c)
        >>> print(pred.shape)  # torch.Size([2, 5, 512])
    """

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        
        # 可学习的位置编码
        # 形状: (1, num_frames, input_dim)
        # 每个位置都有一个learnable的向量
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        
        # 嵌入层Dropout
        self.dropout = nn.Dropout(emb_dropout)
        
        # 条件Transformer：支持动作调制
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        前向传播 - 自回归预测
        
        参数:
            x (Tensor): 上文嵌入序列，形状 (B, T, input_dim)
                       B=批次, T=上文长度, input_dim=嵌入维度
            c (Tensor): 动作编码（条件），形状 (B, action_emb_dim)
        
        返回:
            Tensor: 预测嵌入，形状 (B, T, output_dim)
        
        处理流程:
            1. 添加位置编码：x = x + pos_embedding[:, :T]
            2. 应用Dropout
            3. 通过条件Transformer：基于动作c进行预测
            4. 返回预测嵌入
        
        使用案例:
            >>> predictor = ARPredictor(
            ...     num_frames=20,
            ...     depth=6,
            ...     heads=8,
            ...     mlp_dim=2048,
            ...     input_dim=512,
            ...     hidden_dim=512,
            ...     dim_head=64,
            ...     dropout=0.1,
            ...     emb_dropout=0.1
            ... )
            >>> 
            >>> # Case 1: 单步预测
            >>> ctx_emb = torch.randn(4, 5, 512)      # 上文嵌入 (context_len=5)
            >>> action_emb = torch.randn(4, 512)      # 动作
            >>> pred = predictor(ctx_emb, action_emb)
            >>> print(pred.shape)  # torch.Size([4, 5, 512])
            
            >>> # Case 2: 不同长度上下文
            >>> ctx_emb_10 = torch.randn(4, 10, 512)  # 更长的上文
            >>> pred_10 = predictor(ctx_emb_10, action_emb)
            >>> print(pred_10.shape)  # torch.Size([4, 10, 512])
            
            >>> # Case 3: 批处理
            >>> batch_size, context_len = 32, 8
            >>> ctx = torch.randn(batch_size, context_len, 512)
            >>> action = torch.randn(batch_size, 512)
            >>> predictions = predictor(ctx, action)
            >>> print(predictions.shape)  # torch.Size([32, 8, 512])
            
            >>> # Case 4: 多步预测推理
            >>> # 在eval.py中的rollout()函数中使用
            >>> predictor.eval()
            >>> with torch.no_grad():
            ...     # 迭代预测多个步骤
            ...     context = torch.randn(2, 5, 512)
            ...     for step in range(10):
            ...         action = torch.randn(2, 512)
            ...         next_pred = predictor(context, action)
            ...         # 更新context用于下一步
            ...         context = torch.cat([context[:, 1:], next_pred[:, -1:]], dim=1)
            
            详细流程示例:
            输入: x = (B=2, T=5, dim=512), c = (B=2, dim_c)
            
            1. 添加位置编码:
               x_with_pos = x + pos_embedding[:, :5]
               pos_embedding形状: (1, 5, 512)
               结果: (2, 5, 512)
            
            2. Dropout:
               x_dropped = dropout(x_with_pos)
               (训练时50%的元素随机置0，评估时不变)
            
            3. 条件Transformer:
               - 输入投影 (如需要)
               - 条件投影: c (B, dim_c) → (B, hidden_dim)
               - 6个ConditionalBlock层
               - 每个块: 注意力 + MLP (都由c调制)
               - 最终归一化
               - 输出投影 (如需要)
            
            4. 输出: (2, 5, 512)
        """
        # 获取序列长度
        T = x.size(1)
        
        # 添加位置编码到输入
        # 位置编码是可学习的，初始化为高斯分布
        # shape: (1, T, input_dim)，广播到 (B, T, input_dim)
        x = x + self.pos_embedding[:, :T]
        
        # 应用Dropout到嵌入（有助于正则化）
        x = self.dropout(x)
        
        # 通过条件Transformer
        # x: (B, T, input_dim)
        # c: (B, action_emb_dim)
        # 输出: (B, T, output_dim)
        x = self.transformer(x, c)
        
        return x
