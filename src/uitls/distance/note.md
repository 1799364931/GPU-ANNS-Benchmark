
# Distance模块

在`Faiss`中，distance

# L2 Distance

L2距离在`Faiss`中被拆分成L2范数与内积计算，计算`q`和`v`的距离时，将$|q-x|_{2}$拆分成$|q|_{2} - |x|_{2} + x\cdot q$

