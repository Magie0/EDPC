import torch
from torch import nn

r = 4

class mlpCompressor(torch.nn.Module):
  def __init__(self, vocab_size, vocab_dim,  hidden_dim, n_layers, ffn_dim, n_heads,
               batch_size):
    super(mlpCompressor, self).__init__()

    self.seq_len = 1
    self._vocab_size = vocab_size
    self._hidden_dim = hidden_dim
    
    self.input_map = torch.nn.Embedding(vocab_size, vocab_dim * 2)
    self.output_logit_map = torch.nn.Linear(self._hidden_dim * 2 * self.seq_len, vocab_size)
    
    torch.nn.init.normal_(self.input_map.weight, 0, 0.01)
    torch.nn.init.normal_(self.output_logit_map.weight, 0, 0.01)
    torch.nn.init.zeros_(self.output_logit_map.bias)
    
    self.batch_size = batch_size 
    
    l = []
    l.append(BELayer2(16 * self.seq_len, 16 * 2, 64, batch_size))
    l.append(BELayer11(2 * self.seq_len, 128 * 2, 512 * 4, batch_size))
    l.append(BELayer(2 * self.seq_len, 128 * 2, 64, batch_size))
    
    self.layers = torch.nn.ModuleList(l)

  def forward(self, x, y, last=False):
    x = self.input_map(x)
    
    x = self.layers[0].full_forward(x)
    x = self.layers[1].full_forward(x)
    x = self.layers[2].full_forward(x)

    x = self.output_logit_map(x)

    return x, x
  
  def full_loss(self,
                inputs,
                with_grad=True,
                nonpad_mask=None,
                return_acc=False):
    logits, emb = self.forward(inputs[:, :-1], inputs[:, 1:])
    logits = logits.transpose(1, 2)
    loss = torch.nn.functional.cross_entropy(
            logits[:, :, -1], inputs[:, -1], reduction='mean')
  
    if with_grad:
      loss.backward()

    return loss, logits

class dense_baens0(nn.Module):
  def __init__(self, N=5, B=4, D1=3, D2=2, compression_adjustment=1):
    super(dense_baens0, self).__init__()
    self.N = N
    self.B = B
    self.D1 = D1
    self.D2 = D2
    
    compression_rate = max(1, r - compression_adjustment)
    
    self.linear1 = nn.Linear(D1, D1 // compression_rate, bias=True)
    self.U = nn.Parameter(torch.normal(0, 0.01, (N, D1 // compression_rate, D2 // compression_rate)), requires_grad=True)
    self.linear = nn.Linear(D1 // compression_rate, D2, bias=True)
    self.bias = nn.Parameter(torch.normal(0, 0.01, (N, B, D2)), requires_grad=True)

  def forward(self, x):
    residual = x
    act = self.linear1(x)
    act = torch.bmm(act, self.U)
    act = self.linear(act)
    act += self.bias
    act += residual
    return act

class BELayer2(torch.nn.Module):
  def __init__(self, branch, vocab_dim, ffn_dim, batch_size):
    super(BELayer2, self).__init__()
    self.branch = branch
    self.vocab_dim = vocab_dim
    self.batch_size = batch_size
    
    self.U_map0 = torch.nn.Linear(branch * vocab_dim, 8192, bias=True)
    self.U_map00 = torch.nn.Linear(branch * vocab_dim, 8192, bias=True)
    self.U_map1 = torch.nn.Linear(32, 128, bias=True)
    self.U_map11 = torch.nn.Linear(32, 128, bias=True)
    self.V_map1 = torch.nn.Linear(8192, branch * vocab_dim, bias=True)
    
    self.layernorm11 = torch.nn.LayerNorm(branch * vocab_dim, eps=1e-05, elementwise_affine=True)
    self.ffn_cache = []
  
  def full_forward(self, x):
    x = x.reshape(self.batch_size, 1, self.branch * self.vocab_dim)
    x = self._ffn(x)
    return x
  
  def _ffn(self, x):
    x = self.layernorm11(x)
    skip = x

    if not self.ffn_cache:
      x_main = self.U_map0(x)
      x_main = torch.nn.functional.gelu(x_main)
      x_gate = self.U_map00(skip)
      x = x_main * x_gate
      self.ffn_cache.append(x.detach())
    
    x_last_token = x[:, :, -32:] 
    x1 = x_last_token
    
    x_main = self.U_map1(x_last_token)
    x_main = torch.nn.functional.gelu(x_main)
    x_gate = self.U_map11(x1)
    x_refined = x_main * x_gate
    
    self.ffn_cache[0] = torch.cat((self.ffn_cache[0][:, :, 128:], x_refined), dim=2).detach()
    
    x = self.V_map1(self.ffn_cache[0])
    x = torch.nn.functional.gelu(x)
    
    x = (skip + x) / 2
    return x
  
class BELayer11(torch.nn.Module):
  def __init__(self, branch, vocab_dim, ffn_dim, batch_size):
    super(BELayer11, self).__init__()
    self.branch = branch
    self.vocab_dim = vocab_dim
    self.ffn_dim = ffn_dim
    self.batch_size = batch_size
    
    self.U_map1 = torch.nn.Linear(branch * vocab_dim, ffn_dim, bias=True)
    self.U_map11 = torch.nn.Linear(branch * vocab_dim, ffn_dim, bias=True)
    self.V_map1 = torch.nn.Linear(ffn_dim, branch * vocab_dim, bias=True)
    
    self.layernorm11 = torch.nn.LayerNorm(branch * vocab_dim, eps=1e-05, elementwise_affine=True)
    self.layernorm22 = torch.nn.LayerNorm(branch * vocab_dim, eps=1e-05, elementwise_affine=True)

  def full_forward(self, x):
    x = x.reshape(self.batch_size, 1, self.branch * self.vocab_dim)
    x = self._ffn(x)
    return x
  
  def _ffn(self, x):
    x = self.layernorm11(x)
    skip = x

    x_main = self.U_map1(x)
    x_main = torch.nn.functional.gelu(x_main)
    x_gate = self.U_map11(skip)
    x = x_main * x_gate
    
    x = self.V_map1(x)
    x = self.layernorm22(x)
    x = torch.nn.functional.gelu(x)
    
    x = (skip + x) / 2
    return x

class BELayer(torch.nn.Module):
  def __init__(self, branch, vocab_dim, ffn_dim, batch_size):
    super(BELayer, self).__init__()
    self.branch = branch
    self.vocab_dim = vocab_dim
    self.batch_size = batch_size
    
    self.V_map = dense_baens0(batch_size, branch, vocab_dim, vocab_dim)
    
    self.layernorm1 = torch.nn.LayerNorm(vocab_dim, eps=1e-05, elementwise_affine=True)
    self.layernorm11 = torch.nn.LayerNorm(branch * vocab_dim, eps=1e-05, elementwise_affine=True)
    self.layernorm22 = torch.nn.LayerNorm(branch * vocab_dim, eps=1e-05, elementwise_affine=True)
    
    self.U_map1 = torch.nn.Linear(branch * vocab_dim, 4096, bias=True)
    self.U_map11 = torch.nn.Linear(branch * vocab_dim, 4096, bias=True)
    self.V_map1 = torch.nn.Linear(4096, branch * vocab_dim, bias=True)

  def full_forward(self, x):
    x = x.reshape(self.batch_size, self.branch, self.vocab_dim)
    
    x = self.layernorm1(x)
    skip = x
    x = self.V_map(x)
    x = (skip + x) / 2
    
    x = x.reshape(self.batch_size, 1, self.branch * self.vocab_dim)
    
    x = self._ffn(x)
    return x
  
  def _ffn(self, x):
    x = self.layernorm11(x)
    skip = x

    x_main = self.U_map1(x)
    x_main = torch.nn.functional.gelu(x_main)
    x_gate = self.U_map11(skip)
    x = x_main * x_gate
    
    x = self.V_map1(x)
    x = self.layernorm22(x)
    x = torch.nn.functional.gelu(x)

    x = (skip + x) / 2
    return x