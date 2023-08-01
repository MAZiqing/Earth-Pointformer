import torch
from torch import nn, einsum
from einops import repeat, rearrange
import numpy as np

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# helpers

def exists(val):
    return val is not None


def max_value(t):
    return torch.finfo(t.dtype).max


def batched_index_select(values, indices, dim=1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)


def vector_select_batched_index(values, indices, dim=3, select_size=500):
    B, H, I, neighbor = indices.shape
    B, H, I, D = values.shape
    # select_size = 100
    values_selected = []
    for i in range(0, I, select_size):
        v = repeat(values, 'b h i d -> b h s i d', s=select_size)
        indice = repeat(indices[..., i:i + select_size, :], 'b h s i -> b h s i d', d=D)
        y = v.gather(dim, indice)
        values_selected += [y]
    res = torch.cat(values_selected, dim=2)
    return res


# classes

class MultiheadPointTransformerLayer(nn.Module):
    def __init__(
            self,
            *,
            H,
            W,
            dim,
            device,
            neighbor_r=10,
            dim_pos=2,
            heads=4,
            dim_head=8,
            pos_mlp_hidden_dim=8,
            attn_mlp_hidden_mult=4,
            # select_size=
            # num_neighbors=None
    ):
        super().__init__()
        self.dim_pos = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        # self.num_neighbors = num_neighbors

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.pos_mlp = nn.Sequential(
            nn.Linear(dim_pos, pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, inner_dim)
        )

        attn_inner_dim = inner_dim * attn_mlp_hidden_mult

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(inner_dim, attn_inner_dim, 1, groups=heads),
            nn.ReLU(),
            nn.Conv2d(attn_inner_dim, inner_dim, 1, groups=heads),
        )

        neighbor_r = max(neighbor_r, 2)
        indices = torch.tensor(self.init_neighbor_indices(H=H, W=W, r=neighbor_r))
        # self.indices = torch.tensor(indices).to(device)
        self.register_buffer('indices', indices)

    def init_neighbor_indices(self, H, W, r=10):

        # Create a squared tensor
        tensor = np.random.rand(H, W)

        # Set the radius of the circle
        r = r
        num_within_r = min(int(r**2 * np.pi), H*W)

        # Create a meshgrid of i and j indices
        i, j = np.meshgrid(np.arange(tensor.shape[0]), np.arange(tensor.shape[1]))

        # Flatten the i and j indices
        indices = np.ravel_multi_index((i, j), dims=tensor.shape)

        # Initialize an empty list to store the indices of the points within the circle
        points_within_circle = []

        # Loop over each point in the flattened indices
        for center_index in range(tensor.size):
            # Calculate the distance between the center point and all other points
            distance = np.sqrt((i - i.flat[center_index]) ** 2 + (j - j.flat[center_index]) ** 2)

            # Create a boolean array where the value is True if the distance is less than or equal to r
            idx = np.argpartition(distance.flatten(), num_within_r)
            within_circle = indices.flatten()[idx[:num_within_r]]

            # Append the indices of the points where the boolean array is True to the list
            points_within_circle.append(within_circle)

        # Convert the list of indices to a 2D NumPy array
        points_within_circle = np.asarray(points_within_circle)

        return points_within_circle

    def forward(self, x, mask=None):
        B, n, D = x.shape
        h = self.heads

        # get queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # split out heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # use subtraction of queries to keys. i suppose this is a better inductive bias for point clouds than dot product



        # prepare mask

        # if exists(mask):
        #     mask = rearrange(mask, 'b i -> b i 1') * rearrange(mask, 'b j -> b 1 j')

        # expand values


        indices = self.indices
        indices_with_heads = repeat(indices, 'i j -> b h i j', b=B, h=h)

        # 加速后的方法
        v1 = vector_select_batched_index(v, indices_with_heads)
        # 原方法
        # v2 = repeat(v, 'b h j d -> b h i j d', i=n)
        # v2 = batched_index_select(v2, indices_with_heads, dim=3)

        # 加速后的方法
        k1 = vector_select_batched_index(k, indices_with_heads)
        qk_rel = rearrange(q, 'b h i d -> b h i 1 d') - k1
        # 原方法
        # qk_rel2 = rearrange(q, 'b h i d -> b h i 1 d') - rearrange(k, 'b h j d -> b h 1 j d')
        # qk_rel2 = batched_index_select(qk_rel2, indices_with_heads, dim=3)

        # rel_pos_emb = batched_index_select(rel_pos_emb, indices_with_heads, dim=3)

        # if exists(mask):
        #     mask = batched_index_select(mask, indices, dim=2)

        # add relative positional embeddings to value

        v1 = v1 # + rel_pos_emb

        # use attention mlp, making sure to add relative positional embedding first

        attn_mlp_input = qk_rel # + rel_pos_emb
        attn_mlp_input = rearrange(attn_mlp_input, 'b h i j d -> b (h d) i j')

        sim = self.attn_mlp(attn_mlp_input)

        # masking

        # if exists(mask):
        #     mask_value = -max_value(sim)
        #     mask = rearrange(mask, 'b i j -> b 1 i j')
        #     sim.masked_fill_(~mask, mask_value)

        # attention

        attn = sim.softmax(dim=-2)

        # aggregate

        v1 = rearrange(v1, 'b h i j d -> b i j (h d)')
        agg = einsum('b d i j, b i j d -> b i d', attn, v1)

        # combine heads

        return self.to_out(agg)


if __name__ == '__main__':
    model = MultiheadPointTransformerLayer(H=8, W=8, dim=32, neighbor_r=3, pos_mlp_hidden_dim=8, attn_mlp_hidden_mult=4)
    x = torch.randn(1, 8 * 8, 32)
    pos = torch.randn(1, 8 * 8, 2)
    out = model.forward(x, pos)
    a = 1
