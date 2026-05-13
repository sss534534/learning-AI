"""
AI数学知识库可视化生成器 - SVG版本
生成高质量的SVG数学概念可视化图表
"""

import os

def create_attention_mechanism_svg():
    """创建注意力机制可视化SVG"""
    
    svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 1200" width="1600" height="1200">
  <defs>
    <linearGradient id="blueGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#4299e1"/>
      <stop offset="100%" style="stop-color:#3182ce"/>
    </linearGradient>
    <linearGradient id="orangeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#ed8936"/>
      <stop offset="100%" style="stop-color:#dd6b20"/>
    </linearGradient>
    <linearGradient id="greenGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#38a169"/>
      <stop offset="100%" style="stop-color:#2f855a"/>
    </linearGradient>
    <linearGradient id="purpleGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#805ad5"/>
      <stop offset="100%" style="stop-color:#6b46c1"/>
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="2" dy="2" stdDeviation="3" flood-opacity="0.3"/>
    </filter>
  </defs>
  
  <!-- 背景 -->
  <rect width="1600" height="1200" fill="#f7fafc"/>
  
  <!-- 标题 -->
  <text x="800" y="50" text-anchor="middle" font-family="Arial" font-size="28" font-weight="bold" fill="#1a365d">
    注意力机制 (Attention Mechanism) 可视化
  </text>
  
  <!-- 第一行：QKV计算示意 -->
  
  <!-- 输入序列 -->
  <g transform="translate(50, 100)">
    <text x="200" y="0" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold" fill="#1a365d">
      输入序列
    </text>
    
    <!-- Token圆圈 -->
    <circle cx="80" cy="60" r="35" fill="url(#blueGrad)" stroke="#1a365d" stroke-width="2"/>
    <text x="80" y="65" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="white">0</text>
    <text x="80" y="110" text-anchor="middle" font-family="Arial" font-size="14" fill="#1a365d">The</text>
    
    <circle cx="180" cy="60" r="35" fill="url(#purpleGrad)" stroke="#1a365d" stroke-width="2"/>
    <text x="180" y="65" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="white">1</text>
    <text x="180" y="110" text-anchor="middle" font-family="Arial" font-size="14" fill="#1a365d">cat</text>
    
    <circle cx="280" cy="60" r="35" fill="url(#orangeGrad)" stroke="#1a365d" stroke-width="2"/>
    <text x="280" y="65" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="white">2</text>
    <text x="280" y="110" text-anchor="middle" font-family="Arial" font-size="14" fill="#1a365d">sat</text>
    
    <circle cx="380" cy="60" r="35" fill="url(#greenGrad)" stroke="#1a365d" stroke-width="2"/>
    <text x="380" y="65" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="white">3</text>
    <text x="380" y="110" text-anchor="middle" font-family="Arial" font-size="14" fill="#1a365d">on</text>
  </g>
  
  <!-- Q, K, V 矩阵 -->
  <g transform="translate(600, 80)">
    <!-- Q矩阵 -->
    <rect x="0" y="20" width="200" height="140" rx="10" fill="#fff5f5" stroke="#ed8936" stroke-width="2"/>
    <text x="100" y="10" text-anchor="middle" font-family="Arial" font-size="16" font-weight="bold" fill="#ed8936">
      Query (Q)
    </text>
    <text x="100" y="90" text-anchor="middle" font-family="Courier New" font-size="14" fill="#1a365d">
      Q = W_Q · X
    </text>
    
    <!-- K矩阵 -->
    <rect x="220" y="20" width="200" height="140" rx="10" fill="#f0fff4" stroke="#38a169" stroke-width="2"/>
    <text x="320" y="10" text-anchor="middle" font-family="Arial" font-size="16" font-weight="bold" fill="#38a169">
      Key (K)
    </text>
    <text x="320" y="90" text-anchor="middle" font-family="Courier New" font-size="14" fill="#1a365d">
      K = W_K · X
    </text>
    
    <!-- V矩阵 -->
    <rect x="440" y="20" width="200" height="140" rx="10" fill="#faf5ff" stroke="#805ad5" stroke-width="2"/>
    <text x="540" y="10" text-anchor="middle" font-family="Arial" font-size="16" font-weight="bold" fill="#805ad5">
      Value (V)
    </text>
    <text x="540" y="90" text-anchor="middle" font-family="Courier New" font-size="14" fill="#1a365d">
      V = W_V · X
    </text>
  </g>
  
  <!-- 第二行：注意力计算 -->
  
  <!-- QK^T 矩阵 -->
  <g transform="translate(50, 320)">
    <text x="200" y="0" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold" fill="#1a365d">
      QK^T 注意力分数矩阵
    </text>
    
    <!-- 热力图 -->
    <g transform="translate(0, 20)">
      <!-- 颜色根据数值变化 -->
      <rect x="50" y="0" width="40" height="40" fill="#c6f6d5"/>
      <rect x="90" y="0" width="40" height="40" fill="#9ae6b4"/>
      <rect x="130" y="0" width="40" height="40" fill="#68d391"/>
      <rect x="170" y="0" width="40" height="40" fill="#f6e05e"/>
      
      <rect x="50" y="40" width="40" height="40" fill="#9ae6b4"/>
      <rect x="90" y="40" width="40" height="40" fill="#c6f6d5"/>
      <rect x="130" y="40" width="40" height="40" fill="#9ae6b4"/>
      <rect x="170" y="40" width="40" height="40" fill="#f6e05e"/>
      
      <rect x="50" y="80" width="40" height="40" fill="#68d391"/>
      <rect x="90" y="80" width="40" height="40" fill="#9ae6b4"/>
      <rect x="130" y="80" width="40" height="40" fill="#c6f6d5"/>
      <rect x="170" y="80" width="40" height="40" fill="#fbd38d"/>
      
      <rect x="50" y="120" width="40" height="40" fill="#f6e05e"/>
      <rect x="90" y="120" width="40" height="40" fill="#f6e05e"/>
      <rect x="130" y="120" width="40" height="40" fill="#fbd38d"/>
      <rect x="170" y="120" width="40" height="40" fill="#c6f6d5"/>
      
      <!-- 数值标签 -->
      <text x="70" y="25" text-anchor="middle" font-family="Arial" font-size="12" fill="#1a365d">0.9</text>
      <text x="110" y="25" text-anchor="middle" font-family="Arial" font-size="12" fill="#1a365d">0.7</text>
      <text x="150" y="25" text-anchor="middle" font-family="Arial" font-size="12" fill="#1a365d">0.5</text>
      <text x="190" y="25" text-anchor="middle" font-family="Arial" font-size="12" fill="#1a365d">0.3</text>
      
      <!-- 轴标签 -->
      <text x="270" y="90" text-anchor="middle" font-family="Arial" font-size="12" fill="#718096">Key向量</text>
      <text x="100" y="200" text-anchor="middle" font-family="Arial" font-size="12" fill="#718096">Query向量</text>
      
      <!-- 轴标签具体值 -->
      <text x="70" y="180" text-anchor="middle" font-family="Arial" font-size="11" fill="#1a365d">Q₁</text>
      <text x="110" y="180" text-anchor="middle" font-family="Arial" font-size="11" fill="#1a365d">Q₂</text>
      <text x="150" y="180" text-anchor="middle" font-family="Arial" font-size="11" fill="#1a365d">Q₃</text>
      <text x="190" y="180" text-anchor="middle" font-family="Arial" font-size="11" fill="#1a365d">Q₄</text>
      
      <text x="70" y="195" text-anchor="middle" font-family="Arial" font-size="11" fill="#1a365d">K₁</text>
      <text x="110" y="195" text-anchor="middle" font-family="Arial" font-size="11" fill="#1a365d">K₂</text>
      <text x="150" y="195" text-anchor="middle" font-family="Arial" font-size="11" fill="#1a365d">K₃</text>
      <text x="190" y="195" text-anchor="middle" font-family="Arial" font-size="11" fill="#1a365d">K₄</text>
    </g>
    
    <!-- 公式 -->
    <rect x="300" y="60" width="280" height="80" rx="8" fill="white" stroke="#e2e8f0" stroke-width="1"/>
    <text x="440" y="90" text-anchor="middle" font-family="Courier New" font-size="14" fill="#1a365d">
      Attention(Q,K,V) = softmax(QK^T/√d) · V
    </text>
    <text x="440" y="120" text-anchor="middle" font-family="Courier New" font-size="12" fill="#718096">
      其中 d = Q/K 向量的维度
    </text>
  </g>
  
  <!-- Softmax归一化 -->
  <g transform="translate(700, 320)">
    <text x="300" y="0" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold" fill="#1a365d">
      Softmax 归一化
    </text>
    
    <text x="50" y="50" font-family="Arial" font-size="14" fill="#1a365d">
      αᵢ = exp(scoreᵢ) / Σⱼexp(scoreⱼ)
    </text>
    
    <!-- 归一化后的权重条形图 -->
    <rect x="50" y="80" width="200" height="30" rx="5" fill="#ed8936"/>
    <text x="260" y="100" font-family="Arial" font-size="14" fill="#1a365d">0.45</text>
    
    <rect x="50" y="120" width="140" height="30" rx="5" fill="#38a169"/>
    <text x="200" y="140" font-family="Arial" font-size="14" fill="#1a365d">0.32</text>
    
    <rect x="50" y="160" width="80" height="30" rx="5" fill="#805ad5"/>
    <text x="140" y="180" font-family="Arial" font-size="14" fill="#1a365d">0.18</text>
    
    <rect x="50" y="200" width="25" height="30" rx="5" fill="#718096"/>
    <text x="85" y="220" font-family="Arial" font-size="14" fill="#1a365d">0.05</text>
  </g>
  
  <!-- 第三行：多头注意力 -->
  <g transform="translate(50, 650)">
    <text x="700" y="0" text-anchor="middle" font-family="Arial" font-size="20" font-weight="bold" fill="#1a365d">
      多头注意力 (Multi-Head Attention)
    </text>
    
    <!-- 输入 -->
    <rect x="50" y="50" width="120" height="80" rx="10" fill="url(#blueGrad)" filter="url(#shadow)"/>
    <text x="110" y="85" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="white">
      输入序列
    </text>
    <text x="110" y="110" text-anchor="middle" font-family="Arial" font-size="12" fill="white">
      Input
    </text>
    
    <!-- 箭头 -->
    <line x1="170" y1="90" x2="250" y2="90" stroke="#1a365d" stroke-width="3"/>
    <polygon points="250,90 240,85 240,95" fill="#1a365d"/>
    
    <!-- 多个注意力头 -->
    <g transform="translate(260, 0)">
      <!-- Head 1 -->
      <rect x="0" y="20" width="120" height="60" rx="8" fill="#fed7d7" stroke="#ed8936" stroke-width="2"/>
      <text x="60" y="55" text-anchor="middle" font-family="Arial" font-size="13" font-weight="bold" fill="#ed8936">
        Head₁
      </text>
      
      <!-- Head 2 -->
      <rect x="0" y="100" width="120" height="60" rx="8" fill="#c6f6d5" stroke="#38a169" stroke-width="2"/>
      <text x="60" y="135" text-anchor="middle" font-family="Arial" font-size="13" font-weight="bold" fill="#38a169">
        Head₂
      </text>
      
      <!-- Head 3 -->
      <rect x="0" y="180" width="120" height="60" rx="8" fill="#e9d8fd" stroke="#805ad5" stroke-width="2"/>
      <text x="60" y="215" text-anchor="middle" font-family="Arial" font-size="13" font-weight="bold" fill="#805ad5">
        Head₃
      </text>
      
      <!-- Head 4 -->
      <rect x="0" y="260" width="120" height="60" rx="8" fill="#b2f5ea" stroke="#319795" stroke-width="2"/>
      <text x="60" y="295" text-anchor="middle" font-family="Arial" font-size="13" font-weight="bold" fill="#319795">
        Head₄
      </text>
      
      <!-- 连接线 -->
      <line x1="120" y1="50" x2="220" y2="50" stroke="#ed8936" stroke-width="2"/>
      <line x1="120" y1="130" x2="220" y2="130" stroke="#38a169" stroke-width="2"/>
      <line x1="120" y1="210" x2="220" y2="210" stroke="#805ad5" stroke-width="2"/>
      <line x1="120" y1="290" x2="220" y2="290" stroke="#319795" stroke-width="2"/>
      
      <!-- 汇聚到Concat -->
      <line x1="220" y1="50" x2="270" y2="170" stroke="#718096" stroke-width="1" stroke-dasharray="5,5"/>
      <line x1="220" y1="130" x2="270" y2="170" stroke="#718096" stroke-width="1" stroke-dasharray="5,5"/>
      <line x1="220" y1="210" x2="270" y2="170" stroke="#718096" stroke-width="1" stroke-dasharray="5,5"/>
      <line x1="220" y1="290" x2="270" y2="170" stroke="#718096" stroke-width="1" stroke-dasharray="5,5"/>
    </g>
    
    <!-- Concat -->
    <rect x="270" y="110" width="140" height="120" rx="10" fill="#4299e1" filter="url(#shadow)"/>
    <text x="340" y="160" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="white">
      Concat
    </text>
    <text x="340" y="190" text-anchor="middle" font-family="Arial" font-size="12" fill="white">
      [h₁; h₂; h₃; h₄]
    </text>
    
    <!-- 箭头到Linear -->
    <line x1="410" y1="170" x2="490" y2="170" stroke="#1a365d" stroke-width="3"/>
    <polygon points="490,170 480,165 480,175" fill="#1a365d"/>
    
    <!-- Linear -->
    <rect x="500" y="120" width="120" height="100" rx="10" fill="#1a365d" filter="url(#shadow)"/>
    <text x="560" y="165" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="white">
      Linear
    </text>
    <text x="560" y="195" text-anchor="middle" font-family="Arial" font-size="12" fill="#a0aec0">
      W⁰
    </text>
    
    <!-- 箭头到输出 -->
    <line x1="620" y1="170" x2="700" y2="170" stroke="#1a365d" stroke-width="3"/>
    <polygon points="700,170 690,165 690,175" fill="#1a365d"/>
    
    <!-- 输出 -->
    <rect x="710" y="130" width="120" height="80" rx="10" fill="url(#purpleGrad)" filter="url(#shadow)"/>
    <text x="770" y="165" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="white">
      输出序列
    </text>
    <text x="770" y="190" text-anchor="middle" font-family="Arial" font-size="12" fill="white">
      Output
    </text>
  </g>
  
  <!-- 公式区域 -->
  <g transform="translate(50, 1000)">
    <rect x="0" y="0" width="1500" height="150" rx="10" fill="white" stroke="#e2e8f0" stroke-width="1"/>
    
    <text x="750" y="35" text-anchor="middle" font-family="Arial" font-size="16" font-weight="bold" fill="#1a365d">
      核心公式
    </text>
    
    <text x="200" y="80" text-anchor="middle" font-family="Courier New" font-size="14" fill="#1a365d">
      MultiHead(Q,K,V) = Concat(head₁, ..., headₕ) · W⁰
    </text>
    
    <text x="700" y="80" text-anchor="middle" font-family="Courier New" font-size="14" fill="#38a169">
      headᵢ = Attention(QWᵢᵠ, KWᵢᵏ, VWᵢᵛ)
    </text>
    
    <text x="1200" y="80" text-anchor="middle" font-family="Courier New" font-size="14" fill="#805ad5">
      h = 注意力头数量 (通常为8或16)
    </text>
    
    <text x="750" y="120" text-anchor="middle" font-family="Arial" font-size="12" fill="#718096">
      其中 Wᵢᵠ, Wᵢᵏ, Wᵢᵛ 是可学习的投影矩阵，W⁰ 是输出投影矩阵
    </text>
  </g>
  
</svg>'''
    
    return svg

def create_transformer_svg():
    """创建Transformer架构SVG"""
    
    svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1800 1400" width="1800" height="1400">
  <defs>
    <linearGradient id="blueGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#4299e1"/>
      <stop offset="100%" style="stop-color:#3182ce"/>
    </linearGradient>
    <linearGradient id="orangeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#ed8936"/>
      <stop offset="100%" style="stop-color:#dd6b20"/>
    </linearGradient>
    <linearGradient id="greenGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#38a169"/>
      <stop offset="100%" style="stop-color:#2f855a"/>
    </linearGradient>
    <linearGradient id="purpleGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#805ad5"/>
      <stop offset="100%" style="stop-color:#6b46c1"/>
    </linearGradient>
    <linearGradient id="cyanGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#319795"/>
      <stop offset="100%" style="stop-color:#2c7a7b"/>
    </linearGradient>
  </defs>
  
  <!-- 背景 -->
  <rect width="1800" height="1400" fill="#f7fafc"/>
  
  <!-- 标题 -->
  <text x="900" y="50" text-anchor="middle" font-family="Arial" font-size="28" font-weight="bold" fill="#1a365d">
    Transformer 架构详解
  </text>
  
  <!-- 编码器部分 -->
  <g transform="translate(50, 100)">
    <text x="350" y="0" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold" fill="#1a365d">
      编码器 (Encoder)
    </text>
    <text x="350" y="30" text-anchor="middle" font-family="Arial" font-size="14" fill="#718096">
      N × 层
    </text>
    
    <!-- 输入嵌入 -->
    <rect x="50" y="80" width="150" height="100" rx="10" fill="url(#blueGrad)" filter="url(#shadow)"/>
    <text x="125" y="125" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="white">
      输入嵌入
    </text>
    <text x="125" y="155" text-anchor="middle" font-family="Arial" font-size="12" fill="#e2e8f0">
      Input Embedding
    </text>
    
    <!-- 位置编码 -->
    <rect x="250" y="80" width="150" height="100" rx="10" fill="url(#orangeGrad)" filter="url(#shadow)"/>
    <text x="325" y="125" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="white">
      位置编码
    </text>
    <text x="325" y="155" text-anchor="middle" font-family="Arial" font-size="12" fill="#fff5f5">
      Positional Encoding
    </text>
    
    <!-- 加法 -->
    <text x="445" y="135" text-anchor="middle" font-family="Arial" font-size="28" font-weight="bold" fill="#1a365d">
      +
    </text>
    
    <!-- 编码器Layer -->
    <g transform="translate(500, 60)">
      <!-- Multi-Head Self-Attention -->
      <rect x="0" y="20" width="200" height="100" rx="10" fill="#c6f6d5" stroke="#38a169" stroke-width="2"/>
      <text x="100" y="60" text-anchor="middle" font-family="Arial" font-size="13" font-weight="bold" fill="#38a169">
        Multi-Head
      </text>
      <text x="100" y="85" text-anchor="middle" font-family="Arial" font-size="13" font-weight="bold" fill="#38a169">
        Self-Attention
      </text>
      <text x="100" y="105" text-anchor="middle" font-family="Arial" font-size="11" fill="#718096">
        MHA
      </text>
      
      <!-- Add & Norm 1 -->
      <rect x="0" y="130" width="200" height="50" rx="8" fill="url(#cyanGrad)"/>
      <text x="100" y="160" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold" fill="white">
        + Add &amp; LayerNorm
      </text>
      
      <!-- Feed Forward -->
      <rect x="0" y="190" width="200" height="70" rx="10" fill="#e9d8fd" stroke="#805ad5" stroke-width="2"/>
      <text x="100" y="225" text-anchor="middle" font-family="Arial" font-size="13" font-weight="bold" fill="#805ad5">
        Feed Forward
      </text>
      <text x="100" y="245" text-anchor="middle" font-family="Arial" font-size="11" fill="#718096">
        FFN
      </text>
      
      <!-- Add & Norm 2 -->
      <rect x="0" y="270" width="200" height="50" rx="8" fill="url(#cyanGrad)"/>
      <text x="100" y="300" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold" fill="white">
        + Add &amp; LayerNorm
      </text>
      
      <!-- 连接箭头 -->
      <line x1="100" y1="120" x2="100" y2="130" stroke="#1a365d" stroke-width="2"/>
      <polygon points="100,130 95,120 105,120" fill="#1a365d"/>
      
      <line x1="100" y1="180" x2="100" y2="190" stroke="#1a365d" stroke-width="2"/>
      <polygon points="100,190 95,180 105,180" fill="#1a365d"/>
      
      <line x1="100" y1="260" x2="100" y2="270" stroke="#1a365d" stroke-width="2"/>
      <polygon points="100,270 95,260 105,260" fill="#1a365d"/>
    </g>
    
    <!-- 输出 -->
    <line x1="700" y1="185" x2="800" y2="185" stroke="#1a365d" stroke-width="3"/>
    <polygon points="800,185 790,180 790,190" fill="#1a365d"/>
    
    <rect x="810" y="135" width="150" height="100" rx="10" fill="url(#purpleGrad)" filter="url(#shadow)"/>
    <text x="885" y="175" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="white">
      编码器输出
    </text>
    <text x="885" y="205" text-anchor="middle" font-family="Arial" font-size="12" fill="#e9d8fd">
      Encoder Output
    </text>
  </g>
  
  <!-- 解码器部分 -->
  <g transform="translate(50, 750)">
    <text x="600" y="0" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold" fill="#1a365d">
      解码器 (Decoder)
    </text>
    <text x="600" y="30" text-anchor="middle" font-family="Arial" font-size="14" fill="#718096">
      N × 层
    </text>
    
    <!-- 输出嵌入 -->
    <rect x="50" y="80" width="150" height="100" rx="10" fill="url(#purpleGrad)" filter="url(#shadow)"/>
    <text x="125" y="125" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="white">
      输出嵌入
    </text>
    <text x="125" y="155" text-anchor="middle" font-family="Arial" font-size="12" fill="#e9d8fd">
      Output Embedding
    </text>
    
    <!-- 位置编码 -->
    <rect x="250" y="80" width="150" height="100" rx="10" fill="url(#orangeGrad)" filter="url(#shadow)"/>
    <text x="325" y="125" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="white">
      位置编码
    </text>
    
    <!-- 加法 -->
    <text x="445" y="135" text-anchor="middle" font-family="Arial" font-size="28" font-weight="bold" fill="#1a365d">
      +
    </text>
    
    <!-- 解码器Layer -->
    <g transform="translate(500, 60)">
      <!-- Masked Multi-Head Self-Attention -->
      <rect x="0" y="20" width="200" height="80" rx="10" fill="#fed7d7" stroke="#e53e3e" stroke-width="2"/>
      <text x="100" y="55" text-anchor="middle" font-family="Arial" font-size="13" font-weight="bold" fill="#e53e3e">
        Masked MHA
      </text>
      <text x="100" y="80" text-anchor="middle" font-family="Arial" font-size="11" fill="#718096">
        (防止看到未来)
      </text>
      
      <!-- Add & Norm 1 -->
      <rect x="0" y="110" width="200" height="45" rx="8" fill="url(#cyanGrad)"/>
      <text x="100" y="138" text-anchor="middle" font-family="Arial" font-size="11" font-weight="bold" fill="white">
        + Add &amp; LayerNorm
      </text>
      
      <!-- Cross Attention -->
      <rect x="0" y="165" width="200" height="70" rx="10" fill="#c6f6d5" stroke="#38a169" stroke-width="2"/>
      <text x="100" y="195" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold" fill="#38a169">
        Cross Attention
      </text>
      <text x="100" y="220" text-anchor="middle" font-family="Arial" font-size="10" fill="#718096">
        (关注编码器输出)
      </text>
      
      <!-- Add & Norm 2 -->
      <rect x="0" y="245" width="200" height="45" rx="8" fill="url(#cyanGrad)"/>
      <text x="100" y="273" text-anchor="middle" font-family="Arial" font-size="11" font-weight="bold" fill="white">
        + Add &amp; LayerNorm
      </text>
      
      <!-- Feed Forward -->
      <rect x="0" y="300" width="200" height="60" rx="10" fill="#e9d8fd" stroke="#805ad5" stroke-width="2"/>
      <text x="100" y="330" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold" fill="#805ad5">
        Feed Forward
      </text>
      
      <!-- Add & Norm 3 -->
      <rect x="0" y="370" width="200" height="45" rx="8" fill="url(#cyanGrad)"/>
      <text x="100" y="398" text-anchor="middle" font-family="Arial" font-size="11" font-weight="bold" fill="white">
        + Add &amp; LayerNorm
      </text>
      
      <!-- 连接箭头 -->
      <line x1="100" y1="100" x2="100" y2="110" stroke="#1a365d" stroke-width="2"/>
      <line x1="100" y1="155" x2="100" y2="165" stroke="#1a365d" stroke-width="2"/>
      <line x1="100" y1="235" x2="100" y2="245" stroke="#1a365d" stroke-width="2"/>
      <line x1="100" y1="290" x2="100" y2="300" stroke="#1a365d" stroke-width="2"/>
      <line x1="100" y1="360" x2="100" y2="370" stroke="#1a365d" stroke-width="2"/>
    </g>
    
    <!-- Cross Attention 连接线 -->
    <path d="M 960 260 Q 1100 260 1100 185" stroke="#38a169" stroke-width="2" fill="none" stroke-dasharray="5,5"/>
    
    <!-- 输出层 -->
    <line x1="700" y1="392" x2="800" y2="392" stroke="#1a365d" stroke-width="3"/>
    <polygon points="800,392 790,387 790,397" fill="#1a365d"/>
    
    <!-- Linear -->
    <rect x="810" y="340" width="150" height="100" rx="10" fill="#1a365d" filter="url(#shadow)"/>
    <text x="885" y="385" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="white">
      Linear
    </text>
    <text x="885" y="415" text-anchor="middle" font-family="Arial" font-size="12" fill="#a0aec0">
      投影到词表
    </text>
    
    <!-- Softmax -->
    <rect x="1000" y="340" width="150" height="100" rx="10" fill="url(#purpleGrad)" filter="url(#shadow)"/>
    <text x="1075" y="385" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="white">
      Softmax
    </text>
    <text x="1075" y="415" text-anchor="middle" font-family="Arial" font-size="12" fill="#e9d8fd">
      概率分布
    </text>
    
    <!-- 箭头 -->
    <line x1="960" y1="392" x2="810" y2="392" stroke="#1a365d" stroke-width="3"/>
    <polygon points="810,392 820,387 820,397" fill="#1a365d"/>
    
    <line x1="1075" y1="440" x2="1075" y2="460" stroke="#1a365d" stroke-width="3"/>
    <polygon points="1075,460 1070,450 1080,450" fill="#1a365d"/>
  </g>
  
  <!-- 公式区域 -->
  <g transform="translate(50, 1250)">
    <rect x="0" y="0" width="1700" height="120" rx="10" fill="white" stroke="#e2e8f0" stroke-width="1"/>
    
    <text x="200" y="40" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="#1a365d">
      位置编码
    </text>
    <text x="200" y="70" text-anchor="middle" font-family="Courier New" font-size="12" fill="#1a365d">
      PE(pos,2i) = sin(pos/10000^{2i/d}) 
    </text>
    <text x="200" y="90" text-anchor="middle" font-family="Courier New" font-size="12" fill="#1a365d">
      PE(pos,2i+1) = cos(pos/10000^{2i/d})
    </text>
    
    <text x="600" y="40" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="#38a169">
      Scaled Dot-Product Attention
    </text>
    <text x="600" y="75" text-anchor="middle" font-family="Courier New" font-size="14" fill="#38a169">
      Attention(Q,K,V) = softmax(QK^T / √d_k)V
    </text>
    
    <text x="1050" y="40" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="#805ad5">
      Layer Normalization
    </text>
    <text x="1050" y="75" text-anchor="middle" font-family="Courier New" font-size="14" fill="#805ad5">
      LN(x) = γ ⊙ (x-μ)/√(σ²+ε) + β
    </text>
    
    <text x="1450" y="40" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="#319795">
      Feed Forward
    </text>
    <text x="1450" y="75" text-anchor="middle" font-family="Courier New" font-size="14" fill="#319795">
      FFN(x) = max(0, xW₁+b₁)W₂ + b₂
    </text>
  </g>
  
</svg>'''
    
    return svg

def main():
    """生成所有SVG可视化"""
    output_dir = r'e:\workspace\学习知识库\AI数学知识库\visualizations'
    
    # 创建目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("开始生成SVG可视化图表...")
    
    # 1. 注意力机制
    print("1. 生成注意力机制可视化...")
    svg = create_attention_mechanism_svg()
    with open(os.path.join(output_dir, '01-attention-mechanism.svg'), 'w', encoding='utf-8') as f:
        f.write(svg)
    print("   保存: 01-attention-mechanism.svg")
    
    # 2. Transformer架构
    print("2. 生成Transformer架构可视化...")
    svg = create_transformer_svg()
    with open(os.path.join(output_dir, '02-transformer-architecture.svg'), 'w', encoding='utf-8') as f:
        f.write(svg)
    print("   保存: 02-transformer-architecture.svg")
    
    print("\nSVG可视化图表生成完成！")
    print(f"输出目录: {output_dir}")

if __name__ == "__main__":
    main()
