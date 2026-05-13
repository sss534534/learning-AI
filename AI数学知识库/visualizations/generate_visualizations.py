"""
AI数学知识库可视化生成器
生成高质量的数学概念可视化图表
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义专业配色方案
COLORS = {
    'primary': '#1a365d',      # 深蓝色 - 主要元素
    'secondary': '#2c5282',    # 中蓝色
    'accent': '#ed8936',       # 暖橙色 - 注意力权重
    'success': '#38a169',       # 绿色 - 正向
    'warning': '#e53e3e',       # 红色 - 负向
    'purple': '#805ad5',        # 紫色 - 概率分布
    'cyan': '#319795',          # 青色 - 辅助
    'gray': '#718096',          # 灰色 - 背景
    'light': '#e2e8f0',         # 浅灰色
    'white': '#ffffff',
    'gradient_start': '#4299e1',
    'gradient_end': '#9f7aea',
}

def save_figure(fig, filepath, dpi=300, bbox_inches='tight'):
    """保存图表为高分辨率图片"""
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches,
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"图表已保存: {filepath}")

# ============================================================================
# 1. 注意力机制可视化
# ============================================================================

def create_attention_mechanism_visualization():
    """创建注意力机制可视化 - 展示QKV计算和注意力权重"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)
    
    # 标题
    fig.suptitle('注意力机制 (Attention Mechanism) 可视化', 
                 fontsize=18, fontweight='bold', color=COLORS['primary'], y=0.98)
    
    # ========== 第一行：QKV计算 ==========
    # 输入向量
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('输入序列', fontsize=12, fontweight='bold')
    
    # 绘制输入token
    tokens = ['The', 'cat', 'sat', 'on']
    colors = [COLORS['gradient_start'], COLORS['gradient_end'], 
              COLORS['accent'], COLORS['success']]
    for i, (token, color) in enumerate(zip(tokens, colors)):
        circle = Circle((0.2 + i*0.2, 0.6), 0.08, 
                       facecolor=color, edgecolor=COLORS['primary'], linewidth=2)
        ax1.add_patch(circle)
        ax1.text(0.2 + i*0.2, 0.6, str(i), ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
        ax1.text(0.2 + i*0.2, 0.35, token, ha='center', va='center', fontsize=9)
    
    # 箭头指向
    ax1.annotate('', xy=(0.15, 0.15), xytext=(0.85, 0.15),
                arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))
    ax1.text(0.5, 0.05, 'Token序列', ha='center', fontsize=10, style='italic')
    
    # Q, K, V 向量生成
    for idx, (title, color) in enumerate([('Query (Q)', COLORS['accent']), 
                                           ('Key (K)', COLORS['success']), 
                                           ('Value (V)', COLORS['purple'])]):
        ax = fig.add_subplot(gs[0, idx + 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(title, fontsize=12, fontweight='bold', color=color)
        
        # 绘制向量矩阵
        matrix_size = 4
        cell_width = 0.2
        for i in range(matrix_size):
            for j in range(matrix_size):
                # 对角线颜色更深
                weight = 0.8 if i == j else 0.3 + np.random.random() * 0.2
                rect = patches.Rectangle((0.1 + j*cell_width, 0.85 - i*0.2), 
                                        cell_width - 0.02, 0.15,
                                        linewidth=1, edgecolor='white',
                                        facecolor=mcolors.to_rgba(color, alpha=weight))
                ax.add_patch(rect)
                ax.text(0.1 + j*cell_width + cell_width/2 - 0.01, 
                       0.85 - i*0.2 + 0.075, f'{np.random.uniform(0.1, 0.9):.1f}',
                       ha='center', va='center', fontsize=7, color='white' if weight > 0.5 else 'black')
        
        ax.text(0.5, 0.05, f'{matrix_size}×{matrix_size}', ha='center', 
               fontsize=9, style='italic', color=COLORS['gray'])
    
    # ========== 第二行：注意力分数计算 ==========
    ax_attn = fig.add_subplot(gs[1, :2])
    ax_attn.set_title('注意力分数计算: Attention(Q, K, V) = softmax(QK^T / √d) × V', 
                     fontsize=11, fontweight='bold', pad=10)
    
    # 绘制QK^T矩阵
    matrix = np.random.rand(4, 4)
    matrix = (matrix + matrix.T) / 2  # 对称化
    np.fill_diagonal(matrix, np.random.uniform(0.8, 1.0, 4))  # 对角线更强
    
    im = ax_attn.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # 添加数值标签
    for i in range(4):
        for j in range(4):
            text = ax_attn.text(j, i, f'{matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=9)
    
    ax_attn.set_xticks(range(4))
    ax_attn.set_xticklabels(['K₁', 'K₂', 'K₃', 'K₄'])
    ax_attn.set_yticks(range(4))
    ax_attn.set_yticklabels(['Q₁', 'Q₂', 'Q₃', 'Q₄'])
    ax_attn.set_xlabel('Key向量')
    ax_attn.set_ylabel('Query向量')
    
    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax_attn, shrink=0.8)
    cbar.set_label('注意力分数', fontsize=10)
    
    # ========== Softmax归一化 ==========
    ax_softmax = fig.add_subplot(gs[1, 2:])
    ax_softmax.set_title('Softmax归一化: αᵢ = exp(scoreᵢ) / Σⱼexp(scoreⱼ)', 
                        fontsize=11, fontweight='bold', pad=10)
    
    # 绘制归一化后的注意力权重
    attention_weights = np.random.dirichlet(np.ones(4))
    
    bars = ax_softmax.barh(range(4), attention_weights, 
                           color=[COLORS['accent'], COLORS['success'], 
                                 COLORS['purple'], COLORS['cyan']])
    
    for i, (bar, weight) in enumerate(zip(bars, attention_weights)):
        ax_softmax.text(weight + 0.02, i, f'{weight:.3f}', 
                       va='center', fontsize=10, fontweight='bold')
    
    ax_softmax.set_yticks(range(4))
    ax_softmax.set_yticklabels([f'Query {i+1}对所有Key的注意力' for i in range(4)])
    ax_softmax.set_xlim(0, 1.3)
    ax_softmax.set_xlabel('归一化注意力权重')
    ax_softmax.grid(axis='x', alpha=0.3)
    
    # ========== 第三行：多头注意力示意 ==========
    ax_mha = fig.add_subplot(gs[2, :])
    ax_mha.set_title('多头注意力 (Multi-Head Attention) - 多个注意力头并行计算', 
                    fontsize=12, fontweight='bold', pad=10)
    ax_mha.set_xlim(0, 16)
    ax_mha.set_ylim(0, 6)
    ax_mha.axis('off')
    
    # 输入
    input_box = FancyBboxPatch((0.2, 2.5), 1.5, 1, 
                               boxstyle="round,pad=0.05",
                               facecolor=COLORS['light'], 
                               edgecolor=COLORS['primary'], linewidth=2)
    ax_mha.add_patch(input_box)
    ax_mha.text(0.95, 3, '输入\n序列', ha='center', va='center', 
               fontsize=10, fontweight='bold')
    
    # 多个注意力头
    n_heads = 4
    head_colors = [COLORS['accent'], COLORS['success'], COLORS['purple'], COLORS['cyan']]
    
    for h in range(n_heads):
        y_pos = 4.5 - h * 1.2
        
        # 注意力头
        head_box = FancyBboxPatch((3, y_pos - 0.4), 2, 0.8,
                                  boxstyle="round,pad=0.05",
                                  facecolor=mcolors.to_rgba(head_colors[h], 0.3),
                                  edgecolor=head_colors[h], linewidth=2)
        ax_mha.add_patch(head_box)
        ax_mha.text(4, y_pos, f'Head {h+1}', ha='center', va='center',
                   fontsize=10, fontweight='bold', color=head_colors[h])
        
        # 连接线
        ax_mha.annotate('', xy=(3, y_pos), xytext=(1.7, 3),
                       arrowprops=dict(arrowstyle='->', color=head_colors[h], lw=1.5))
    
    # Concatenation
    concat_box = FancyBboxPatch((6.5, 2.2), 1.8, 1.6,
                                boxstyle="round,pad=0.05",
                                facecolor=COLORS['gradient_start'],
                                edgecolor=COLORS['primary'], linewidth=2)
    ax_mha.add_patch(concat_box)
    ax_mha.text(7.4, 3, 'Concat\n[h₁;h₂;h₃;h₄]', ha='center', va='center',
               fontsize=9, fontweight='bold', color='white')
    
    # 连接线到concat
    for h in range(n_heads):
        y_pos = 4.5 - h * 1.2
        ax_mha.annotate('', xy=(6.5, 3), xytext=(5, y_pos),
                       arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1))
    
    # 线性变换
    linear_box = FancyBboxPatch((9, 2.5), 1.5, 1,
                               boxstyle="round,pad=0.05",
                               facecolor=COLORS['primary'],
                               edgecolor=COLORS['primary'], linewidth=2)
    ax_mha.add_patch(linear_box)
    ax_mha.text(9.75, 3, 'Linear\nW⁰', ha='center', va='center',
               fontsize=9, fontweight='bold', color='white')
    
    # 连接
    ax_mha.annotate('', xy=(9, 3), xytext=(8.3, 3),
                   arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))
    
    # 输出
    output_box = FancyBboxPatch((11.2, 2.5), 1.5, 1,
                                boxstyle="round,pad=0.05",
                                facecolor=COLORS['light'],
                                edgecolor=COLORS['primary'], linewidth=2)
    ax_mha.add_patch(output_box)
    ax_mha.text(11.95, 3, '输出\n序列', ha='center', va='center',
               fontsize=10, fontweight='bold')
    
    ax_mha.annotate('', xy=(11.2, 3), xytext=(10.5, 3),
                   arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))
    
    # 标注公式
    ax_mha.text(14, 3, 'MultiHead(Q,K,V) = Concat(head₁,...,headₕ)W⁰\n\nheadᵢ = Attention(QWᵢᵠ, KWᵢᵏ, VWᵢᵛ)',
               fontsize=9, ha='left', va='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return fig

# ============================================================================
# 2. Transformer架构可视化
# ============================================================================

def create_transformer_architecture_visualization():
    """创建Transformer架构可视化"""
    
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.2, 1], hspace=0.3)
    
    fig.suptitle('Transformer 架构详解', fontsize=20, fontweight='bold', 
                color=COLORS['primary'], y=0.98)
    
    # ========== 上半部分：编码器-解码器结构 ==========
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 编码器部分 (左侧)
    encoder_x = 1.5
    
    # 输入嵌入 + 位置编码
    input_box = FancyBboxPatch((encoder_x, 8), 2, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=COLORS['gradient_start'],
                               edgecolor=COLORS['primary'], linewidth=2)
    ax.add_patch(input_box)
    ax.text(encoder_x+1, 8.75, '输入嵌入', ha='center', va='center', 
           fontsize=11, fontweight='bold', color='white')
    ax.text(encoder_x+1, 8.25, 'Input Embedding', ha='center', va='center', 
           fontsize=8, color='white')
    
    # 加法符号
    ax.text(encoder_x+2.8, 8.75, '+', fontsize=20, ha='center', va='center')
    
    # 位置编码
    pos_box = FancyBboxPatch((encoder_x+3, 8), 1.5, 1.5,
                             boxstyle="round,pad=0.1",
                             facecolor=COLORS['accent'],
                             edgecolor=COLORS['primary'], linewidth=2)
    ax.add_patch(pos_box)
    ax.text(encoder_x+3.75, 8.75, '位置编码', ha='center', va='center', 
           fontsize=10, fontweight='bold', color='white')
    ax.text(encoder_x+3.75, 8.25, 'Positional Enc.', ha='center', va='center', 
           fontsize=8, color='white')
    
    ax.annotate('', xy=(encoder_x+3, 8.75), xytext=(encoder_x+3, 8.75),
               arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))
    
    # Nx编码器层
    encoder_layer_x = encoder_x + 5.5
    for layer in range(3):
        layer_y = 7 - layer * 2.2
        
        # Multi-Head Self-Attention
        attn_box = FancyBboxPatch((encoder_layer_x, layer_y), 3, 1.2,
                                  boxstyle="round,pad=0.08",
                                  facecolor=mcolors.to_rgba(COLORS['success'], 0.2),
                                  edgecolor=COLORS['success'], linewidth=2)
        ax.add_patch(attn_box)
        ax.text(encoder_layer_x+1.5, layer_y+0.6, 'Multi-Head\nSelf-Attention', 
               ha='center', va='center', fontsize=9, fontweight='bold', color=COLORS['success'])
        ax.text(encoder_layer_x+1.5, layer_y+0.15, 'MHA', ha='center', va='center', 
               fontsize=7, color=COLORS['gray'])
        
        # Add & Norm 1
        norm1_box = FancyBboxPatch((encoder_layer_x, layer_y-0.8), 3, 0.6,
                                   boxstyle="round,pad=0.05",
                                   facecolor=COLORS['cyan'],
                                   edgecolor=COLORS['primary'], linewidth=1.5)
        ax.add_patch(norm1_box)
        ax.text(encoder_layer_x+1.5, layer_y-0.5, '+Norm', ha='center', va='center', 
               fontsize=8, fontweight='bold', color='white')
        
        # Feed Forward
        ff_box = FancyBboxPatch((encoder_layer_x, layer_y-1.6), 3, 0.6,
                                boxstyle="round,pad=0.05",
                                facecolor=mcolors.to_rgba(COLORS['purple'], 0.2),
                                edgecolor=COLORS['purple'], linewidth=2)
        ax.add_patch(ff_box)
        ax.text(encoder_layer_x+1.5, layer_y-1.3, 'Feed Forward', ha='center', va='center', 
               fontsize=9, fontweight='bold', color=COLORS['purple'])
        
        # Add & Norm 2
        norm2_box = FancyBboxPatch((encoder_layer_x, layer_y-2.4), 3, 0.6,
                                   boxstyle="round,pad=0.05",
                                   facecolor=COLORS['cyan'],
                                   edgecolor=COLORS['primary'], linewidth=1.5)
        ax.add_patch(norm2_box)
        ax.text(encoder_layer_x+1.5, layer_y-2.1, '+Norm', ha='center', va='center', 
               fontsize=8, fontweight='bold', color='white')
        
        # 连接箭头
        if layer < 2:
            ax.annotate('', xy=(encoder_layer_x+1.5, layer_y-2.8), 
                       xytext=(encoder_layer_x+1.5, layer_y-2.4),
                       arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.5))
    
    # 编码器标注
    ax.add_patch(FancyBboxPatch((encoder_layer_x-0.3, 6.6), 3.6, 5.5,
                                boxstyle="round,pad=0.15",
                                facecolor='none',
                                edgecolor=COLORS['primary'], linewidth=2, linestyle='--'))
    ax.text(encoder_layer_x+1.5, 12.3, '×N', fontsize=16, fontweight='bold', 
           color=COLORS['primary'], ha='center')
    ax.text(encoder_layer_x+1.5, 12, '编码器 (Encoder)', fontsize=12, fontweight='bold', 
           color=COLORS['primary'], ha='center')
    
    # ========== 解码器部分 (右侧) ==========
    decoder_x = 11
    
    # 输出嵌入
    output_emb_box = FancyBboxPatch((decoder_x, 8), 2, 1.5,
                                    boxstyle="round,pad=0.1",
                                    facecolor=COLORS['gradient_end'],
                                    edgecolor=COLORS['primary'], linewidth=2)
    ax.add_patch(output_emb_box)
    ax.text(decoder_x+1, 8.75, '输出嵌入', ha='center', va='center', 
           fontsize=11, fontweight='bold', color='white')
    
    # 位置编码
    ax.add_patch(FancyBboxPatch((decoder_x+2.5, 8), 1.5, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor=COLORS['accent'],
                                edgecolor=COLORS['primary'], linewidth=2))
    ax.text(decoder_x+3.25, 8.75, '位置编码', ha='center', va='center', 
           fontsize=10, fontweight='bold', color='white')
    
    # 解码器层
    decoder_layer_x = decoder_x + 4.5
    for layer in range(3):
        layer_y = 7 - layer * 2.2
        
        # Masked Multi-Head Self-Attention
        masked_attn_box = FancyBboxPatch((decoder_layer_x, layer_y), 2.5, 1.2,
                                         boxstyle="round,pad=0.08",
                                         facecolor=mcolors.to_rgba(COLORS['warning'], 0.2),
                                         edgecolor=COLORS['warning'], linewidth=2)
        ax.add_patch(masked_attn_box)
        ax.text(decoder_layer_x+1.25, layer_y+0.6, 'Masked MHA', 
               ha='center', va='center', fontsize=9, fontweight='bold', color=COLORS['warning'])
        
        # Norm 1
        ax.add_patch(FancyBboxPatch((decoder_layer_x, layer_y-0.8), 2.5, 0.6,
                                    boxstyle="round,pad=0.05",
                                    facecolor=COLORS['cyan'],
                                    edgecolor=COLORS['primary'], linewidth=1.5))
        ax.text(decoder_layer_x+1.25, layer_y-0.5, '+Norm', ha='center', va='center', 
               fontsize=8, fontweight='bold', color='white')
        
        # Cross Attention
        cross_box = FancyBboxPatch((decoder_layer_x, layer_y-1.6), 2.5, 0.6,
                                  boxstyle="round,pad=0.05",
                                  facecolor=mcolors.to_rgba(COLORS['success'], 0.2),
                                  edgecolor=COLORS['success'], linewidth=2)
        ax.add_patch(cross_box)
        ax.text(decoder_layer_x+1.25, layer_y-1.3, 'Cross Attention', ha='center', va='center', 
               fontsize=8, fontweight='bold', color=COLORS['success'])
        
        # Norm 2
        ax.add_patch(FancyBboxPatch((decoder_layer_x, layer_y-2.4), 2.5, 0.6,
                                    boxstyle="round,pad=0.05",
                                    facecolor=COLORS['cyan'],
                                    edgecolor=COLORS['primary'], linewidth=1.5))
        ax.text(decoder_layer_x+1.25, layer_y-2.1, '+Norm', ha='center', va='center', 
               fontsize=8, fontweight='bold', color='white')
        
        # 连接箭头
        if layer < 2:
            ax.annotate('', xy=(decoder_layer_x+1.25, layer_y-2.8), 
                       xytext=(decoder_layer_x+1.25, layer_y-2.4),
                       arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.5))
    
    # 解码器标注
    ax.add_patch(FancyBboxPatch((decoder_layer_x-0.3, 6.6), 2.9, 5.5,
                                boxstyle="round,pad=0.15",
                                facecolor='none',
                                edgecolor=COLORS['primary'], linewidth=2, linestyle='--'))
    ax.text(decoder_layer_x+1.15, 12.3, '×N', fontsize=16, fontweight='bold', 
           color=COLORS['primary'], ha='center')
    ax.text(decoder_layer_x+1.15, 12, '解码器 (Decoder)', fontsize=12, fontweight='bold', 
           color=COLORS['primary'], ha='center')
    
    # Cross Attention连接线
    for layer in range(3):
        layer_y = 7 - layer * 2.2
        ax.annotate('', xy=(decoder_layer_x, layer_y-1.3), 
                   xytext=(encoder_layer_x+3, layer_y-2.1),
                   arrowprops=dict(arrowstyle='->', color=COLORS['success'], 
                                 lw=1, linestyle='dashed', alpha=0.6))
    
    # 输出层
    output_linear = FancyBboxPatch((decoder_layer_x+3, 5.5), 2, 1.5,
                                  boxstyle="round,pad=0.1",
                                  facecolor=COLORS['primary'],
                                  edgecolor=COLORS['primary'], linewidth=2)
    ax.add_patch(output_linear)
    ax.text(decoder_layer_x+4, 6.25, 'Linear', ha='center', va='center', 
           fontsize=10, fontweight='bold', color='white')
    
    output_softmax = FancyBboxPatch((decoder_layer_x+3, 3.8), 2, 1.5,
                                   boxstyle="round,pad=0.1",
                                   facecolor=COLORS['purple'],
                                   edgecolor=COLORS['primary'], linewidth=2)
    ax.add_patch(output_softmax)
    ax.text(decoder_layer_x+4, 4.55, 'Softmax', ha='center', va='center', 
           fontsize=10, fontweight='bold', color='white')
    
    ax.annotate('', xy=(decoder_layer_x+4, 5.5), xytext=(decoder_layer_x+4, 5.3),
               arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))
    ax.annotate('', xy=(decoder_layer_x+4, 3.8), xytext=(decoder_layer_x+4, 3.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))
    
    # 输出概率
    output_box = FancyBboxPatch((decoder_layer_x+3, 2), 2, 1.2,
                               boxstyle="round,pad=0.1",
                               facecolor=COLORS['light'],
                               edgecolor=COLORS['primary'], linewidth=2)
    ax.add_patch(output_box)
    ax.text(decoder_layer_x+4, 2.6, '输出概率', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    ax.annotate('', xy=(decoder_layer_x+4, 2), xytext=(decoder_layer_x+4, 1.8),
               arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))
    
    # ========== 下半部分：关键组件公式 ==========
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 18)
    ax2.set_ylim(0, 6)
    ax2.axis('off')
    ax2.set_title('核心组件数学形式', fontsize=14, fontweight='bold', pad=10)
    
    # 位置编码公式
    formulas = [
        ('位置编码 (Positional Encoding)',
         r'$PE_{(pos,2i)} = sin(pos/10000^{2i/d_{model}})$\n$PE_{(pos,2i+1)} = cos(pos/10000^{2i/d_{model}})$',
         COLORS['accent']),
        ('Scaled Dot-Product Attention',
         r'$\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$',
         COLORS['success']),
        ('Layer Normalization',
         r'$\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$',
         COLORS['purple']),
        ('Feed Forward',
         r'$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$',
         COLORS['cyan']),
    ]
    
    for idx, (title, formula, color) in enumerate(formulas):
        x_pos = 1 + idx * 4.3
        
        box = FancyBboxPatch((x_pos, 0.8), 4, 4.5,
                            boxstyle="round,pad=0.15",
                            facecolor=mcolors.to_rgba(color, 0.1),
                            edgecolor=color, linewidth=2)
        ax2.add_patch(box)
        
        ax2.text(x_pos+2, 4.8, title, ha='center', va='center', 
                fontsize=10, fontweight='bold', color=color)
        ax2.text(x_pos+2, 2.8, formula, ha='center', va='center', 
                fontsize=11, style='italic',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return fig

# ============================================================================
# 3. 梯度下降可视化
# ============================================================================

def create_gradient_descent_visualization():
    """创建梯度下降优化过程可视化"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    fig.suptitle('梯度下降优化过程 (Gradient Descent)', fontsize=18, fontweight='bold', 
                color=COLORS['primary'], y=0.98)
    
    # ========== 第一行：不同学习率的效果 ==========
    learning_rates = [0.05, 0.15, 0.3]
    lr_names = ['学习率过小\n(收敛慢)', '学习率适中\n(理想)', '学习率过大\n(振荡/发散)']
    lr_colors = [COLORS['cyan'], COLORS['success'], COLORS['warning']]
    
    # 定义一个复杂的损失函数地形
    def loss_function(x, y):
        return np.sin(x * 0.5) * np.cos(y * 0.5) * 0.5 + \
               (x ** 2 + y ** 2) * 0.05
    
    def grad_loss(x, y):
        dx = np.cos(x * 0.5) * 0.5 * 0.5 + x * 0.1
        dy = -np.sin(y * 0.5) * 0.5 * 0.5 + y * 0.1
        return dx, dy
    
    for idx, (lr, name, color) in enumerate(zip(learning_rates, lr_names, lr_colors)):
        ax = fig.add_subplot(gs[0, idx])
        
        # 创建等高线
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        Z = loss_function(X, Y)
        
        contour = ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
        
        # 梯度下降路径
        path_x = [-4]
        path_y = [-4]
        x_current, y_current = -4, -4
        
        for step in range(50):
            dx, dy = grad_loss(x_current, y_current)
            x_new = x_current - lr * dx
            y_new = y_current - lr * dy
            
            # 检查是否发散
            if np.abs(x_new) > 10 or np.abs(y_new) > 10:
                break
            
            path_x.append(x_new)
            path_y.append(y_new)
            x_current, y_current = x_new, y_new
        
        # 绘制路径
        ax.plot(path_x, path_y, 'r-', linewidth=2, alpha=0.7, label='优化路径')
        ax.scatter(path_x, path_y, c=range(len(path_x)), cmap='Reds', 
                  s=50, zorder=5, edgecolors='black', linewidths=0.5)
        ax.scatter(path_x[-1], path_y[-1], c='lime', s=200, marker='*', 
                  zorder=10, edgecolors='black', linewidths=1, label='最优点')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(name, fontsize=11, fontweight='bold', color=color)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(-5.5, 5.5)
        ax.set_ylim(-5.5, 5.5)
        ax.grid(True, alpha=0.3)
    
    # ========== 第二行左：优化器对比 ==========
    ax_compare = fig.add_subplot(gs[1, :2])
    ax_compare.set_title('不同优化器收敛速度对比', fontsize=14, fontweight='bold', pad=10)
    
    # 生成不同优化器的收敛曲线
    steps = np.arange(0, 200)
    
    # SGD
    loss_sgd = 1 / (steps + 1) ** 0.3 + np.random.randn(200) * 0.02
    ax_compare.plot(steps, loss_sgd, label='SGD', color=COLORS['gray'], linewidth=2)
    
    # SGD + Momentum
    loss_momentum = 1 / (steps + 1) ** 0.5 + np.random.randn(200) * 0.015
    ax_compare.plot(steps, loss_momentum, label='SGD + Momentum', 
                   color=COLORS['cyan'], linewidth=2)
    
    # Adam
    loss_adam = 1 / (steps + 1) ** 0.8 + np.random.randn(200) * 0.01
    ax_compare.plot(steps, loss_adam, label='Adam', color=COLORS['success'], linewidth=2.5)
    
    # AdamW
    loss_adamw = 1 / (steps + 1) ** 0.85 + np.random.randn(200) * 0.008
    ax_compare.plot(steps, loss_adamw, label='AdamW', color=COLORS['accent'], linewidth=2.5)
    
    # LAMB
    loss_lamb = 1 / (steps + 1) ** 0.9 + np.random.randn(200) * 0.005
    ax_compare.plot(steps, loss_lamb, label='LAMB', color=COLORS['purple'], linewidth=2.5)
    
    ax_compare.set_xlabel('训练步数', fontsize=12)
    ax_compare.set_ylabel('Loss', fontsize=12)
    ax_compare.legend(loc='upper right', fontsize=10)
    ax_compare.grid(True, alpha=0.3)
    ax_compare.set_yscale('log')
    ax_compare.set_ylim(0.05, 1.2)
    
    # ========== 第二行右：Adam更新公式 ==========
    ax_formula = fig.add_subplot(gs[1, 2])
    ax_formula.set_xlim(0, 1)
    ax_formula.set_ylim(0, 1)
    ax_formula.axis('off')
    ax_formula.set_title('Adam优化器更新公式', fontsize=12, fontweight='bold', pad=10)
    
    formulas_adam = [
        ('梯度估计 (一阶矩)', r'$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$'),
        ('二阶矩估计', r'$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$'),
        ('偏差校正', r'$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \hat{v}_t = \frac{v_t}{1-\beta_2^t}$'),
        ('参数更新', r'$\theta_{t+1} = \theta_t - \frac{\alpha \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$'),
    ]
    
    for idx, (title, formula) in enumerate(formulas_adam):
        y_pos = 0.85 - idx * 0.22
        
        ax_formula.text(0.5, y_pos, title, ha='center', va='center', 
                       fontsize=10, fontweight='bold', color=COLORS['primary'])
        ax_formula.text(0.5, y_pos - 0.08, formula, ha='center', va='center', 
                       fontsize=11, style='italic',
                       bbox=dict(boxstyle='round', facecolor=COLORS['light'], 
                                edgecolor=COLORS['gray'], alpha=0.8))
    
    # 典型参数
    ax_formula.text(0.5, 0.05, r'$\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$', 
                   ha='center', va='center', fontsize=9, style='italic', color=COLORS['gray'])
    
    return fig

# ============================================================================
# 4. 扩散模型流程可视化
# ============================================================================

def create_diffusion_model_visualization():
    """创建扩散模型前向/逆向过程可视化"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)
    
    fig.suptitle('扩散模型 (Diffusion Model) 流程详解', fontsize=18, fontweight='bold', 
                color=COLORS['primary'], y=0.98)
    
    # ========== 第一行：前向扩散过程 ==========
    ax_forward = fig.add_subplot(gs[0, :])
    ax_forward.set_xlim(0, 16)
    ax_forward.set_ylim(0, 2)
    ax_forward.axis('off')
    ax_forward.set_title('前向扩散过程 (Forward Process) - 逐步添加噪声', 
                        fontsize=13, fontweight='bold', pad=10, color=COLORS['success'])
    
    # 绘制图像演变
    n_steps = 8
    for i in range(n_steps):
        x_pos = 1 + i * 1.8
        t = i / (n_steps - 1)  # 0到1
        
        # 图像框
        img_box = FancyBboxPatch((x_pos, 0.3), 1.2, 1.2,
                                boxstyle="round,pad=0.05",
                                facecolor=plt.cm.Greys(1 - t * 0.7),
                                edgecolor=COLORS['primary'], linewidth=2)
        ax_forward.add_patch(img_box)
        
        # 添加噪声点
        if t > 0.1:
            noise_level = int(t * 30)
            for _ in range(noise_level):
                nx = x_pos + 0.1 + np.random.random() * 1
                ny = 0.4 + np.random.random() * 1
                color = plt.cm.RdYlGn(np.random.random())
                ax_forward.scatter(nx, ny, s=3, c=[color], alpha=0.5)
        
        # 时间步标注
        ax_forward.text(x_pos + 0.6, 1.6, f't={i*100}', ha='center', 
                       fontsize=9, fontweight='bold')
        ax_forward.text(x_pos + 0.6, 0.1, f'{int(t*1000)}', ha='center', 
                       fontsize=8, color=COLORS['gray'])
    
    # 箭头
    for i in range(n_steps - 1):
        x_pos = 1 + i * 1.8
        ax_forward.annotate('', xy=(x_pos + 1.7, 0.9), xytext=(x_pos + 1.25, 0.9),
                           arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2))
    
    # 公式
    ax_forward.text(14.5, 1.5, r'$q(x_t | x_{t-1}) = N(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$',
                   fontsize=10, style='italic',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax_forward.text(14.5, 0.8, r'$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$',
                   fontsize=10, style='italic',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # ========== 第二行：逆向去噪过程 ==========
    ax_reverse = fig.add_subplot(gs[1, :])
    ax_reverse.set_xlim(0, 16)
    ax_reverse.set_ylim(0, 2)
    ax_reverse.axis('off')
    ax_reverse.set_title('逆向去噪过程 (Reverse Process) - 逐步恢复图像', 
                        fontsize=13, fontweight='bold', pad=10, color=COLORS['warning'])
    
    # 从右到左
    for i in range(n_steps):
        x_pos = 14 - i * 1.8
        t = i / (n_steps - 1)  # 0到1
        
        img_box = FancyBboxPatch((x_pos, 0.3), 1.2, 1.2,
                                boxstyle="round,pad=0.05",
                                facecolor=plt.cm.Greys(1 - (1 - t) * 0.7),
                                edgecolor=COLORS['primary'], linewidth=2)
        ax_reverse.add_patch(img_box)
        
        # 清晰度增加
        if t > 0.1:
            for _ in range(int((1-t) * 30)):
                nx = x_pos + 0.1 + np.random.random() * 1
                ny = 0.4 + np.random.random() * 1
                color = plt.cm.RdYlGn(np.random.random())
                ax_reverse.scatter(nx, ny, s=3, c=[color], alpha=0.3 * (1-t))
        
        ax_reverse.text(x_pos + 0.6, 1.6, f't=T-{i*100}', ha='center', 
                       fontsize=9, fontweight='bold')
    
    for i in range(n_steps - 1):
        x_pos = 14 - i * 1.8
        ax_reverse.annotate('', xy=(x_pos - 0.05, 0.9), xytext=(x_pos + 0.55, 0.9),
                           arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2))
    
    # 公式
    ax_reverse.text(1.5, 1.5, r'$p_\theta(x_{t-1}|x_t) = N(\mu_\theta(x_t,t), \Sigma_\theta(x_t,t))$',
                   fontsize=10, style='italic',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # ========== 第三行左：噪声调度 ==========
    ax_schedule = fig.add_subplot(gs[2, :2])
    ax_schedule.set_title('噪声调度 (Noise Schedule) 对比', fontsize=12, fontweight='bold', pad=10)
    
    t_values = np.linspace(0, 1, 1000)
    
    # 线性调度
    beta_linear = 0.0001 + (0.02 - 0.0001) * t_values
    alpha_linear = np.cumprod(1 - beta_linear)
    
    # 余弦调度
    s = 0.008
    alpha_bar_cosine = np.cos((t_values + s) / (1 + s) * np.pi / 2) ** 2
    
    # Sigmoid调度
    beta_sigmoid = 0.0001 + 0.02 / (1 + np.exp(-10 * (t_values - 0.5)))
    alpha_bar_sigmoid = np.cumprod(1 - beta_sigmoid)
    
    ax_schedule.plot(t_values, alpha_linear, label='线性调度 (Linear)', 
                    color=COLORS['warning'], linewidth=2)
    ax_schedule.plot(t_values, alpha_bar_cosine, label='余弦调度 (Cosine)', 
                    color=COLORS['success'], linewidth=2)
    ax_schedule.plot(t_values, alpha_bar_sigmoid, label='Sigmoid调度', 
                    color=COLORS['purple'], linewidth=2)
    
    ax_schedule.set_xlabel('时间步 t (0→T)', fontsize=11)
    ax_schedule.set_ylabel(r'$\bar{\alpha}_t$', fontsize=11)
    ax_schedule.legend(loc='best', fontsize=10)
    ax_schedule.grid(True, alpha=0.3)
    ax_schedule.set_title('噪声调度对比', fontsize=12, fontweight='bold')
    
    # ========== 第三行右：DDPM训练-采样流程 ==========
    ax_flow = fig.add_subplot(gs[2, 2:])
    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 1)
    ax_flow.axis('off')
    ax_flow.set_title('DDPM 训练与采样流程', fontsize=12, fontweight='bold', pad=10)
    
    # 训练流程
    train_box = FancyBboxPatch((0.05, 0.6), 0.4, 0.3,
                               boxstyle="round,pad=0.05",
                               facecolor=COLORS['success'],
                               edgecolor=COLORS['primary'], linewidth=1.5)
    ax_flow.add_patch(train_box)
    ax_flow.text(0.25, 0.75, '训练', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
    
    train_steps = [
        ('1. 采样x₀~真实数据', 0.5),
        ('2. 采样t~Uniform', 0.35),
        ('3. 采样ε~N(0,I)', 0.2),
        ('4. 计算损失: L = ||ε - εθ||²', 0.05),
    ]
    
    for step, (text, y) in enumerate(train_steps):
        ax_flow.text(0.05, y, step + 1, ha='left', va='center', fontsize=8,
                    fontweight='bold', color=COLORS['success'])
        ax_flow.text(0.1, y, text, ha='left', va='center', fontsize=8)
    
    # 采样流程
    sample_box = FancyBboxPatch((0.55, 0.6), 0.4, 0.3,
                               boxstyle="round,pad=0.05",
                               facecolor=COLORS['warning'],
                               edgecolor=COLORS['primary'], linewidth=1.5)
    ax_flow.add_patch(sample_box)
    ax_flow.text(0.75, 0.75, '采样', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
    
    sample_steps = [
        ('1. 采样x_T~N(0,I)', 0.5),
        ('2. For t=T...1:', 0.35),
        ('3. 计算μθ, Σθ', 0.2),
        ('4. 采样x_{t-1}', 0.05),
    ]
    
    for idx, (text, y) in enumerate(sample_steps):
        ax_flow.text(0.55, y, f'{idx+1}', ha='left', va='center', fontsize=8,
                    fontweight='bold', color=COLORS['warning'])
        ax_flow.text(0.6, y, text, ha='left', va='center', fontsize=8)
    
    return fig

# ============================================================================
# 5. MoE架构可视化
# ============================================================================

def create_moe_architecture_visualization():
    """创建MoE混合专家架构可视化"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    fig.suptitle('混合专家模型 (Mixture of Experts, MoE) 架构', fontsize=18, 
                fontweight='bold', color=COLORS['primary'], y=0.98)
    
    # ========== 上左：稀疏激活机制 ==========
    ax_sparse = fig.add_subplot(gs[0, 0])
    ax_sparse.set_xlim(0, 10)
    ax_sparse.set_ylim(0, 8)
    ax_sparse.axis('off')
    ax_sparse.set_title('稀疏激活机制 (Sparse Gating)', fontsize=13, 
                        fontweight='bold', pad=10, color=COLORS['accent'])
    
    # 输入
    input_box = FancyBboxPatch((0.5, 3), 1.5, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor=COLORS['gradient_start'],
                              edgecolor=COLORS['primary'], linewidth=2)
    ax_sparse.add_patch(input_box)
    ax_sparse.text(1.25, 3.75, '输入x', ha='center', va='center', 
                  fontsize=11, fontweight='bold', color='white')
    
    # 路由门控
    gate_box = FancyBboxPatch((2.8, 2.8), 1.8, 2,
                             boxstyle="round,pad=0.1",
                             facecolor=COLORS['accent'],
                             edgecolor=COLORS['primary'], linewidth=2)
    ax_sparse.add_patch(gate_box)
    ax_sparse.text(3.7, 4, '门控网络\nG(x)', ha='center', va='center', 
                  fontsize=10, fontweight='bold', color='white')
    
    ax_sparse.annotate('', xy=(2.8, 3.75), xytext=(2, 3.75),
                      arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))
    
    # 专家网络 (6个)
    n_experts = 6
    k = 2  # top-k
    expert_positions = [(5.5, 1), (7.5, 1), (9, 3), (7.5, 5), (5.5, 6), (3.5, 4.5)]
    expert_active = [True, False, True, False, False, True]  # 假设激活2个
    
    for i, ((ex, ey), active) in enumerate(zip(expert_positions, expert_active)):
        color = COLORS['success'] if active else COLORS['gray']
        alpha = 1.0 if active else 0.4
        
        expert_box = FancyBboxPatch((ex - 0.6, ey - 0.5), 1.2, 1,
                                   boxstyle="round,pad=0.08",
                                   facecolor=mcolors.to_rgba(color, 0.3) if active 
                                            else mcolors.to_rgba(color, 0.15),
                                   edgecolor=color, linewidth=2 if active else 1)
        ax_sparse.add_patch(expert_box)
        ax_sparse.text(ex, ey, f'E{i+1}', ha='center', va='center', 
                      fontsize=10, fontweight='bold', color=color if active else COLORS['gray'])
        
        # 连接线
        if i < 3:  # 连接到门控
            ax_sparse.annotate('', xy=(ex, ey + 0.5), xytext=(4.6, 4),
                             arrowprops=dict(arrowstyle='->', color=color, 
                                           lw=1.5 if active else 0.5, alpha=alpha if active else 0.3))
    
    # 顶部标注
    ax_sparse.text(5, 7.5, f'Top-{k} 选择', ha='center', fontsize=10, 
                  fontweight='bold', color=COLORS['accent'],
                  bbox=dict(boxstyle='round', facecolor=COLORS['light'], edgecolor=COLORS['accent']))
    
    # 输出
    output_box = FancyBboxPatch((0.5, 0.5), 1.5, 1.2,
                               boxstyle="round,pad=0.1",
                               facecolor=COLORS['gradient_end'],
                               edgecolor=COLORS['primary'], linewidth=2)
    ax_sparse.add_patch(output_box)
    ax_sparse.text(1.25, 1.1, '输出y', ha='center', va='center', 
                  fontsize=11, fontweight='bold', color='white')
    
    # 输出连接
    ax_sparse.annotate('', xy=(1.25, 1.7), xytext=(3.5, 3),
                      arrowprops=dict(arrowstyle='->', color=COLORS['gradient_end'], lw=2))
    
    # 公式
    ax_sparse.text(5, 0.3, r'$y = \sum_{i \in TopK(G(x))} G(x)_i \cdot E_i(x)$',
                  fontsize=11, ha='center', style='italic',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # ========== 上右：负载均衡可视化 ==========
    ax_balance = fig.add_subplot(gs[0, 1])
    ax_balance.set_title('负载均衡重要性', fontsize=13, fontweight='bold', pad=10)
    
    # 不均衡的专家使用
    experts = np.arange(1, 9)
    load_imbalanced = np.array([0.4, 0.25, 0.15, 0.1, 0.05, 0.03, 0.015, 0.005])
    load_balanced = np.ones(8) / 8
    
    x = np.arange(len(experts))
    width = 0.35
    
    bars1 = ax_balance.bar(x - width/2, load_imbalanced, width, 
                          label='不均衡', color=COLORS['warning'], alpha=0.7)
    bars2 = ax_balance.bar(x + width/2, load_balanced, width, 
                          label='均衡', color=COLORS['success'], alpha=0.7)
    
    ax_balance.set_xlabel('专家编号', fontsize=11)
    ax_balance.set_ylabel('使用频率', fontsize=11)
    ax_balance.set_xticks(x)
    ax_balance.set_xticklabels([f'E{i}' for i in experts])
    ax_balance.legend(loc='upper right')
    ax_balance.grid(axis='y', alpha=0.3)
    
    # 添加负载均衡损失说明
    ax_balance.text(0.02, 0.95, r'$\mathcal{L}_{aux} = \alpha \sum_i P_i \cdot I_i$',
                   transform=ax_balance.transAxes, fontsize=10, style='italic',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========== 下左：FFN到MoE演进 ==========
    ax_evolution = fig.add_subplot(gs[1, 0])
    ax_evolution.set_xlim(0, 10)
    ax_evolution.set_ylim(0, 6)
    ax_evolution.axis('off')
    ax_evolution.set_title('Dense到MoE的演进', fontsize=13, fontweight='bold', pad=10)
    
    # Dense FFN
    dense_box = FancyBboxPatch((0.5, 4), 3, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=COLORS['primary'],
                               edgecolor=COLORS['primary'], linewidth=2)
    ax_evolution.add_patch(dense_box)
    ax_evolution.text(2, 5, 'Dense FFN', ha='center', va='center', 
                     fontsize=12, fontweight='bold', color='white')
    ax_evolution.text(2, 4.4, '(所有参数激活)', ha='center', va='center', 
                     fontsize=9, color=COLORS['light'])
    
    # Sparse MoE
    sparse_box = FancyBboxPatch((0.5, 1), 3, 2,
                               boxstyle="round,pad=0.1",
                               facecolor=COLORS['accent'],
                               edgecolor=COLORS['primary'], linewidth=2)
    ax_evolution.add_patch(sparse_box)
    ax_evolution.text(2, 2.6, 'Sparse MoE', ha='center', va='center', 
                     fontsize=12, fontweight='bold', color='white')
    ax_evolution.text(2, 2, '(top-k专家激活)', ha='center', va='center', 
                     fontsize=9, color='white')
    
    # 专家展开
    for i, (x_pos, y_pos) in enumerate([(4, 4.2), (4.8, 4.2), (5.6, 4.2), 
                                         (4.4, 3.4), (5.2, 3.4)]):
        expert = FancyBboxPatch((x_pos, y_pos), 0.6, 0.6,
                                boxstyle="round,pad=0.05",
                                facecolor=COLORS['success'],
                                edgecolor=COLORS['primary'], linewidth=1.5)
        ax_evolution.add_patch(expert)
        ax_evolution.text(x_pos + 0.3, y_pos + 0.3, f'E{i+1}', 
                         ha='center', va='center', fontsize=8, 
                         fontweight='bold', color='white')
    
    # 箭头和标注
    ax_evolution.annotate('', xy=(3.5, 2.5), xytext=(3.5, 4.5),
                         arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2))
    ax_evolution.text(4, 3.5, '稀疏\n激活', ha='center', va='center', 
                     fontsize=9, fontweight='bold', color=COLORS['accent'])
    
    # 优势说明
    ax_evolution.text(6.5, 5.2, '优势对比', fontsize=11, fontweight='bold')
    
    advantages = [
        ('参数量', '固定', 'N×E (E=专家数)'),
        ('激活量', '100%', 'K/N × 100%'),
        ('计算量', 'O(N)', 'O(K)'),
        ('显存', '大', '可扩展'),
    ]
    
    for i, (item, dense, moe) in enumerate(advantages):
        y = 4.5 - i * 0.8
        ax_evolution.text(6.5, y, item + ':', fontsize=9, fontweight='bold')
        ax_evolution.text(7.5, y, f'Dense: {dense}', fontsize=8, color=COLORS['gray'])
        ax_evolution.text(8.5, y, f'MoE: {moe}', fontsize=8, color=COLORS['accent'])
    
    # ========== 下右：专家特化示意 ==========
    ax_specialization = fig.add_subplot(gs[1, 1])
    ax_specialization.set_title('专家特化现象', fontsize=13, fontweight='bold', pad=10)
    
    # 创建专家-任务热力图
    tasks = ['代码生成', '数学推理', '文学创作', '对话问答', '翻译']
    experts_names = [f'专家{i}' for i in range(1, 7)]
    
    # 模拟专家特化矩阵
    specialization = np.array([
        [0.85, 0.1, 0.05, 0.0, 0.0],
        [0.1, 0.8, 0.05, 0.05, 0.0],
        [0.0, 0.05, 0.85, 0.05, 0.05],
        [0.05, 0.1, 0.1, 0.7, 0.05],
        [0.0, 0.05, 0.1, 0.15, 0.7],
        [0.05, 0.05, 0.05, 0.1, 0.75],
    ])
    
    im = ax_specialization.imshow(specialization, cmap='YlGn', aspect='auto', vmin=0, vmax=1)
    
    for i in range(len(experts_names)):
        for j in range(len(tasks)):
            text = ax_specialization.text(j, i, f'{specialization[i, j]:.2f}',
                                        ha="center", va="center", color="black", fontsize=8)
    
    ax_specialization.set_xticks(range(len(tasks)))
    ax_specialization.set_xticklabels(tasks, rotation=45, ha='right', fontsize=9)
    ax_specialization.set_yticks(range(len(experts_names)))
    ax_specialization.set_yticklabels(experts_names)
    
    cbar = plt.colorbar(im, ax=ax_specialization, shrink=0.8)
    cbar.set_label('专业程度', fontsize=10)
    
    return fig

# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    import os
    
    # 创建输出目录
    output_dir = r'e:\workspace\学习知识库\AI数学知识库\visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    print("开始生成可视化图表...")
    
    # 1. 注意力机制
    print("1. 生成注意力机制可视化...")
    fig = create_attention_mechanism_visualization()
    save_figure(fig, os.path.join(output_dir, '01-attention-mechanism.png'))
    
    # 2. Transformer架构
    print("2. 生成Transformer架构可视化...")
    fig = create_transformer_architecture_visualization()
    save_figure(fig, os.path.join(output_dir, '02-transformer-architecture.png'))
    
    # 3. 梯度下降
    print("3. 生成梯度下降可视化...")
    fig = create_gradient_descent_visualization()
    save_figure(fig, os.path.join(output_dir, '03-gradient-descent.png'))
    
    # 4. 扩散模型
    print("4. 生成扩散模型可视化...")
    fig = create_diffusion_model_visualization()
    save_figure(fig, os.path.join(output_dir, '04-diffusion-model.png'))
    
    # 5. MoE架构
    print("5. 生成MoE架构可视化...")
    fig = create_moe_architecture_visualization()
    save_figure(fig, os.path.join(output_dir, '05-moe-architecture.png'))
    
    print("\n所有可视化图表生成完成！")
    print(f"输出目录: {output_dir}")
