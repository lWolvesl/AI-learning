# No deep learning,just function mapping

$$
X = [v_1,v_2,.....,v_{784}]\\
X:[1,dx]
$$

$$
H_1 = XW_{1} + b_{1} \\
W_1:[d_1,dx] \\
b_1:[d_1]
$$

$$
H_2 = H_1W_2 + b_2 \\
W_1:[d_2,d_1] \\
b_1:[d_2]
$$

$$
H_3=H_2W_3 + b_3 \\
W_3:[10,d_2]\\
b_3:[10]
$$

## Loss

$$
H_3:[1,d_3] \\
Y:[0/1/2/.../9] \\
eg.:1\geq[0,1,0,0,0,0,0,0,0,0,0] \\
eg.:3\geq[0,0,0,1,0,0,0,0,0,0,0] \\
Euclidean\ Distance:H_3\ vs\ Y
$$



## In a nutshell

$$
pred = W_3 \times \{W_2\cdot[W_1X+b_1]+b_2\}+b_3
$$

