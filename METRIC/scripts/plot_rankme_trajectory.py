#!/usr/bin/env python3
"""Plot task- and dataset-weighted RankMe rankings for 16 checkpoints."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

root=Path(__file__).resolve().parent.parent
out=root/'results/sample20'
data=pd.read_csv(out/'rankme-trajectory16-summary.csv')
data['step']=data.checkpoint.str.removeprefix('step-').astype(int)
data=data.sort_values('step')
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'figure.dpi':150,'savefig.dpi':200,'font.size':11})
fig,ax=plt.subplots(figsize=(13,6.6),layout='constrained')
fig.patch.set_facecolor('#f8fafc');ax.set_facecolor('white')
x=data.step/1000
ax.plot(x,data.task_mean_rank,color='#1663ad',marker='o',markersize=6,linewidth=2.6,label='任务等权（53 个任务）',zorder=3)
ax.plot(x,data.dataset_mean_rank,color='#8898ab',marker='s',markersize=4.5,linewidth=1.8,linestyle='--',label='数据集等权（8 个数据集）',zorder=2)
observed=data[data.step.isin([10000,30000,50000,70000,80000])]
ax.scatter(observed.step/1000,observed.task_mean_rank,s=125,facecolor='white',edgecolor='#1663ad',linewidth=2,zorder=4)
best=data.loc[data.task_mean_rank.idxmax()];ax.scatter([best.step/1000],[best.task_mean_rank],s=170,color='#e57722',edgecolor='white',linewidth=1.3,zorder=5)
ax.annotate(f"最佳：step-{int(best.step)}\n平均排名 {best.task_mean_rank:.2f} / 16",xy=(best.step/1000,best.task_mean_rank),xytext=(46,13.35),color='#a6480d',fontsize=11,weight='bold',arrowprops={'arrowstyle':'->','color':'#c25c17','lw':1.5},bbox={'boxstyle':'round,pad=0.45','facecolor':'#fff4e8','edgecolor':'#f2c18f'})
ax.set_title('FaVoR 预训练权重：RankMe 综合排名',loc='left',fontsize=19,weight='bold',pad=18)
ax.set_xlabel('预训练步数（千步）',labelpad=10);ax.set_ylabel('平均排名（越高越好）',labelpad=10)
ax.set_xticks(x);ax.set_xlim(3,82);ax.set_ylim(0,17);ax.set_yticks([1,4,8,12,16]);ax.grid(axis='y',color='#e4e9ef',linewidth=.8)
for spine in ax.spines.values():spine.set_visible(False)
handles,labels=ax.get_legend_handles_labels();handles.append(Line2D([0],[0],marker='o',markerfacecolor='white',markeredgecolor='#1663ad',markeredgewidth=2,linestyle='None',markersize=8));labels.append('曾参与五权重微调对照')
ax.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.24),fontsize=10)
fig.text(.015,-.045,'八个音视频数据集各抽取训练集＋验证集的 20%；共同有效记录 25,752 条。每个任务先对 16 个权重排名，再汇总。\n任务等权包含 38 个临床任务；step-60000 未参与此前五权重微调对照。',fontsize=9,color='#526171',fontproperties=font)
fig.savefig(out/'rankme-trajectory16.png',bbox_inches='tight',facecolor=fig.get_facecolor())
fig.savefig(out/'rankme-trajectory16.pdf',bbox_inches='tight',facecolor=fig.get_facecolor())
print(out/'rankme-trajectory16.png')
