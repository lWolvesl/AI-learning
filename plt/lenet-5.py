from graphviz import Digraph

# 创建有向图
dot = Digraph(comment='LeNet-5', format='png', 
              graph_attr={'rankdir': 'LR', 'splines': 'line', 'nodesep': '0.5'})

# 定义节点样式
input_style = {'shape': 'box', 'style': 'filled', 'fillcolor': 'lightblue'}
conv_style = {'shape': 'box', 'style': 'filled', 'fillcolor': 'lightgreen'}
pool_style = {'shape': 'box', 'style': 'filled', 'fillcolor': 'lightcoral'}
fc_style = {'shape': 'box', 'style': 'filled', 'fillcolor': 'lightyellow'}
output_style = {'shape': 'box', 'style': 'filled', 'fillcolor': 'lightpink'}

# 添加节点，包含更多详细信息
dot.node('input', 'Input\n32x32x1\nGrayscale', **input_style)
dot.node('C1', 'Conv1\n5x5 kernel\n6 filters\nStride=1\n28x28x6', **conv_style)
dot.node('S2', 'Pool1\n2x2 kernel\nStride=2\n14x14x6', **pool_style)
dot.node('C3', 'Conv2\n5x5 kernel\n16 filters\nStride=1\n10x10x16', **conv_style)
dot.node('S4', 'Pool2\n2x2 kernel\nStride=2\n5x5x16', **pool_style)
dot.node('C5', 'FC1\n120 neurons', **fc_style)
dot.node('F6', 'FC2\n84 neurons', **fc_style)
dot.node('output', 'Output\n10 classes\n(Softmax)', **output_style)

# 添加边，带箭头
dot.edge('input', 'C1', label='Conv')
dot.edge('C1', 'S2', label='MaxPool')
dot.edge('S2', 'C3', label='Conv')
dot.edge('C3', 'S4', label='MaxPool')
dot.edge('S4', 'C5', label='Flatten\n+ FC')
dot.edge('C5', 'F6', label='FC')
dot.edge('F6', 'output', label='Softmax')

# 修改层数标注的显示方式
with dot.subgraph() as s:
    s.attr(rank='same')
    s.node('L1', 'Layer 1', shape='plaintext', pos='0,0!')
    s.node('L2', 'Layer 2', shape='plaintext', pos='1,0!')
    s.node('L3', 'Layer 3', shape='plaintext', pos='2,0!')
    s.node('L4', 'Layer 4', shape='plaintext', pos='3,0!')
    s.node('L5', 'Layer 5', shape='plaintext', pos='4,0!')
    s.node('L6', 'Layer 6', shape='plaintext', pos='5,0!')
    s.node('L7', 'Layer 7', shape='plaintext', pos='6,0!')

# 添加层数标注与模型结构的连接
dot.edge('L1', 'C1', style='invis')
dot.edge('L2', 'S2', style='invis')
dot.edge('L3', 'C3', style='invis')
dot.edge('L4', 'S4', style='invis')
dot.edge('L5', 'C5', style='invis')
dot.edge('L6', 'F6', style='invis')
dot.edge('L7', 'output', style='invis')

# 保存并渲染图像
dot.render('plt/lenet-5', view=False, cleanup=True)