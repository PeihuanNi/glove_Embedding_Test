# 实验Embedding能不能用向量加减法得到目标向量
比如验证：
$$women-man+king=queen$$
## 实验方法和结果
我使用了**glove.6B.100d**数据集，使用了**点积注意力机制**和**余弦相似度**进行评估，挑选topk个数据，k=10
### 点积注意力

- #### 公式
$$Similarity=Softmax(Q \times K^T)$$
                    注意这里的$Similarity$和下面的$Possibility$做对比，两个的含义并不相同
- #### 代码
```py
def dot_attention_max(tensor, matrix):
    distance = torch.matmul(tensor, matrix.T)
    possibility = F.softmax(distance, dim=-1)
    topk_value, topk_idx = torch.topk(possibility, 10)
    return topk_value, topk_idx
```
- #### 实验结果
|Class    |Possibility|
| :-----: | :-------: |
|king     |96.02%     |
|queen    |3.40%      |
|emperor  |0.26%      |
|daughteer|0.11%      |
|throne   |0.05%      |
|princess |0.04%      |
|prince   |0.03%      |
|wife     |0.02%      |
|son      |0.01%      |
|mother   |0.01%      |

可见$DotAttention+Softmax$并不是一个很好的方法，他会极大的抑制这个较小的值

### 余弦相似度
余弦相似度其实和点积注意力本质上相同，但是余弦相似度除以了向量的模，得到的就是向量间的$\cos$，并且由于cos的值域在$(-1, 1)$，所以他就可以作为$Similarity$
- #### 公式
$$Similarity=\frac{Q \times K^T}{|Q|\times |K|}$$
- #### 代码
```py
def dot_attention_max(tensor, matrix):
    distance = torch.matmul(tensor, matrix.T)
    norm_tensor = torch.norm(tensor, dim=0)
    norm_matrix = torch.norm(matrix, dim=1)
    similarity = distance / (norm_matrix * norm_tensor)
    topk_value, topk_idx = torch.topk(similarity, 10)
    return topk_value, topk_idx
```
- #### 实验结果
|Class    |Possibility|
| :-----: | :-------: |
|king     |85.52%     |
|queen    |78.34%     |
|throne   |68.33%     |
|daughter |68.09%     |
|prince   |67.13%     |
|princess |66.44%     |
|mother   |65.79%     |
|elizabeth|65.63%     |
|father   |63.92%     |
可以看到余弦相似度在这里要比点积注意力的效果要好的多，因为没有softmax来增大差异

经过多次实验，使用不同单词发现，相似度最大的总是位于king那个位置的单词，可能这也是为什么主流的transformer架构会在topk（k一般为3）里面随机取一个的原因。