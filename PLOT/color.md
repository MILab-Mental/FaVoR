Unified Color Palette

论文所有图表统一使用以下色卡，并保持全篇颜色语义一致。这些颜色是有顺序的
Primary palette：
#ED66A4, #8690C2, #8CC7A1, #F59E42, #959BA5, #C4B080, #9E7BBC, #F1F1F1

Secondary palette：
#C94F8B, #6673A8, #6FA987, #D17E32, #707780, #A98F5C, #805F9F, #D5D5D5

Color usage rules：

优先使用 Primary palette，尤其是前 4 种颜色用于核心类别、模型、方法或实验组区分（main_result）。
对同一语义类别需要进行深浅区分时，使用对应的 Secondary color。例如：
Primary #ED66A4 → Secondary #C94F8B
Primary #8690C2 → Secondary #6673A8
Secondary palette 不能被当作新的独立类别颜色随意使用，而应作为对应 Primary color 的强化版本。
#959BA5 / #C4B080 / #9E7BBC 主要用于次要类别、baseline、辅助模块、额外实验组等。
#F1F1F1 / #D5D5D5 主要用于背景、网格线、分隔区域和弱强调元素，不要用于需要重点识别的数据类别。
禁止使用额外的随机颜色。若类别超过 8 个，应优先通过明度、线型、纹理、marker、透明度或分组方式进行区分，而不是继续引入新的色相。
所有图表中同一语义对象必须保持完全一致的颜色。例如某模型在 Figure 2 中使用 #ED66A4，则 Figure 5、Figure 6、Supplementary Figures 中也必须使用 #ED66A4。
柱状图、折线图、散点图、violin plot、box plot、heatmap、Sankey diagram 等均遵循该统一颜色语义。
整体视觉风格保持柔和、低饱和、现代、专业、简洁的学术论文风格，适合 Nature / IEEE / MICCAI / Medical AI 类论文。
避免高饱和荧光色、彩虹色、渐变色、霓虹色和无语义装饰性颜色。