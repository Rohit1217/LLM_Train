import torch
from einops import einsum,rearrange
import torch.nn.functional as F


x=torch.randn(3,4,5).to("cuda:5")
dot_einsum=torch.einsum("ijk,ijk->",x,x)
print(dot_einsum)


### DOT PRODUCT ###

a,b=torch.randn(3),torch.randn(3)

print(torch.dot(a,b),torch.einsum("i,i->",a,b))


### TRANSPOSE ###

a=torch.randn(2,3,4,5)
print(torch.einsum("ijkl->ilkj",a).shape)

### MATRIX ELEMMENT WISE PRODUCT SUM
a,b=torch.randn(4,4),torch.randn(4,4)
ab=torch.einsum("ij,ij->",a,b)
print(ab)


### MATRIX OUTER PRODUCT
a,b=torch.randn(6),torch.randn(8)
print(torch.einsum("i,j->ij",a,b).shape)

### DIAGONAL SQUARE MATRIX
a=torch.randn(4,4)
print(a)
print(torch.einsum("ii->i",a))

### MATRIX TRACE 
print(torch.einsum("ii->",a))


### ROW WISE L2 NORM SQUARED
print(torch.einsum("ij,ij->i",a,a))

### COL WISE L2 NORM SQUARED
print(torch.einsum("ij,ij->j",a,a))


###  BATCH DOT PRODUCT
a,b=torch.randn(32,8),torch.randn(32,8)
print(torch.einsum("ij,ij->i",a,b).shape)


### TENSOR CONTRACTION
a,b=torch.randn(3,4,5),torch.randn(4,5,6)
print(torch.einsum("ijk,jkl->il",a,b))


### BILINEAR FORM
u,W,b=torch.randn(4),torch.randn(4,5),torch.randn(5)
print(torch.einsum("i,ij,j->",u,W,b))


###COVARIANCE 
x=torch.randn(100,8)
print(torch.einsum("ni,nj->ij",x,x).shape)

##MHA
att,v=torch.randn(32,4,8,8),torch.randn(32,4,8,16)
print(torch.einsum("bhnk,bhkd->bhnd",att,v).shape)


### MHA ###

qkv=torch.randn(5,4,128*3)
qkv=rearrange(qkv, "Batch Seq (k d)-> Batch Seq k d",k=3)
qkv=rearrange(qkv, "Batch Seq k (nhead d) -> Batch nhead k Seq  d",nhead=4)

q,k,v=qkv.unbind(dim=2)

att=einsum(q,k,"batch nhead seq1 hd, batch nhead seq2 hd -> batch nhead seq1 seq2")
att=F.softmax(att,dim=-1)

out_x=einsum(att,v,"batch nhead seq1 seq2, batch nhead seq2 hd -> batch nhead seq1 hd")
out_x=rearrange(out_x, "batch nhead seq1 hd -> batch seq1 (nhead hd)")

print(out_x.shape)



