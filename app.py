"""
金融AI智能助手 — Web界面
启动: python app.py
"""
import sys, os, re
from datetime import datetime
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import pandas as pd, numpy as np
import gradio as gr
import plotly.graph_objects as go

from agent_v3 import SelfCorrectingAgent
from fraud_detector import FraudDetector
from equity_penetration import EquityGraph
from portfolio import Portfolio, auto_fill
from realtime_data import get_stock_quote, get_market_overview
from event_clustering import EventClusterer

print("[Web] 初始化...")
agent = SelfCorrectingAgent()
fraud = FraudDetector()
equity = EquityGraph()
clusterer = EventClusterer()
fin_data = pd.read_pickle(os.path.join(BASE, "data_processed", "financials_merged.pkl"))
ALL_STOCKS = sorted(fin_data['stock_code'].dropna().unique())

# 简易名称映射
rpt_df = pd.read_pickle(os.path.join(BASE, "data_processed", "reports.pkl"))
CODE2NAME = {}
for _, r in rpt_df[['sec_code','sec_name']].drop_duplicates().iterrows():
    CODE2NAME[str(r['sec_code']).strip()] = str(r['sec_name']).strip()
print(f"[Web] {len(ALL_STOCKS)}只股票就绪")

# === 图表 ===
def gauge_chart(score):
    c = "#22c55e" if score<30 else "#eab308" if score<60 else "#ef4444"
    fig = go.Figure(go.Indicator(mode="gauge+number",value=score,title={"text":"风险评分"},
        gauge={"axis":{"range":[0,100]},"bar":{"color":c},"steps":[{"range":[0,30],"color":"#dcfce7"},{"range":[30,60],"color":"#fef9c3"},{"range":[60,100],"color":"#fee2e2"}]},
        number={"font":{"size":36,"color":c}}))
    fig.update_layout(height=250,margin=dict(t=40,b=10,l=20,r=20))
    return fig

def radar_chart(alerts):
    am={a['rule']:a['score'] for a in alerts}
    cats=["存货","现金流","应收","负债","商誉","毛利"]
    vals=[am.get("存货积压",0)+am.get("存货增速异常",0),am.get("现金流悖离",0)+am.get("持续现金流失血",0)+am.get("利润现金背离",0),am.get("应收款畸高",0)+am.get("应收增速异常",0),am.get("负债过高",0)+am.get("资不抵债",0),am.get("商誉减值风险",0),am.get("毛利率极低",0)+am.get("毛利率骤降",0)]
    mv=max(vals) if max(vals)>0 else 1
    vals=[min(v/mv*100,100) for v in vals]
    fig=go.Figure();fig.add_trace(go.Scatterpolar(r=vals,theta=cats,fill='toself',fillcolor='rgba(239,68,68,0.2)',line=dict(color='#ef4444',width=2)))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0,100],showticklabels=False)),height=300,margin=dict(t=30,b=30,l=40,r=40))
    return fig

def sunburst_chart(ctrls):
    if not ctrls: return go.Figure()
    labels,parents,values=[],[],[]
    for c in ctrls[:10]:
        chain=c['chain'].split(" -> ")
        for i,node in enumerate(chain):
            short=node[:15]+("..." if len(node)>15 else "")
            if short not in labels:labels.append(short);parents.append("" if i==0 else chain[i-1][:15]);values.append(c['effective_pct'])
    fig=go.Figure(go.Sunburst(labels=labels,parents=parents,values=values,textinfo="label+percent entry",maxdepth=3,insidetextorientation='horizontal',textfont=dict(size=14,color='#1e293b'),hovertemplate='<b>%{label}</b><br>有效持股:%{value:.2f}%<extra></extra>'))
    fig.update_layout(height=500,margin=dict(t=10,b=10,l=10,r=10));fig.update_traces(marker=dict(line=dict(color='#fff',width=2)))
    return fig

# === Tab 1: 对话 ===
def chat_fn(message, history):
    if not message.strip(): return history
    reply, meta = agent.chat(message)
    flag = f"[{'深度' if meta['think_flag'] else '普通'}]"
    header = f"### {flag} 意图:{meta['intent']} | {meta['think_reason']}\n\n"
    history.append({"role":"user","content":message})
    history.append({"role":"assistant","content":header+reply})
    return history

# === Tab 2: 快速分析 ===
def quick_analysis(stock_code):
    if not stock_code or len(stock_code.strip())<3: return None,None,None,"*请输入有效股票代码*"
    code=stock_code.strip()
    f=fraud.analyze(code)
    if f.get('error'): return None,None,None,f"未找到 {code}"
    code=f['stock_code']
    ctrls=equity.find_controller(code)
    flags=equity.check_flags(code)
    short=code.replace('.SH','').replace('.SZ','').replace('.BJ','')
    rr=rpt_df[rpt_df['sec_code'].astype(str).str.contains(short,na=False)].head(3)
    rpt_lines=[f"- [{str(r['publish_dt'])[:10]}] {str(r['title'])[:60]}" for _,r in rr.iterrows()]
    rpt="\n".join(rpt_lines) if rpt_lines else "无相关研报"
    q=get_stock_quote(code)
    qt=""
    if q.get('status')=='ok': qt=f"\n实时:**{q['price']:.2f}**({q['change_pct']:+.2f}%) | {q['name']}"
    text=f"## {code} 综合分析{qt}\n\n### 财务健康度:{f['risk_score']:.0f}/100 [{f['risk_level']}]\n\n"
    if f['alerts']:
        text+=f"**{f['alert_count']}项预警:**\n\n"
        for a in f['alerts']: text+=f"- **{a['rule']}**(+{a['score']:.0f}):{a['detail']}\n  *{a['data']}*\n"
    text+=f"\n### 股权结构\n"
    for c in ctrls[:5]: text+=f"- [{'个人' if c['controller_type']=='个人' else '企业'}] {c['chain']}(有效{c['effective_pct']:.2f}%)\n"
    if flags.get('flags'):
        text+="\n### 股权风险\n"
        for fl in flags['flags']: text+=f"- {fl['flag']}:{fl['detail']}\n"
    text+=f"\n### 相关研报\n\n{rpt}"
    return gauge_chart(f['risk_score']),radar_chart(f['alerts']),sunburst_chart(ctrls),text

# === Tab 3: 风险排行 ===
_RISK_CACHE=None

def risk_ranking(min_score=50):
    global _RISK_CACHE
    if _RISK_CACHE is not None:
        df=_RISK_CACHE[_RISK_CACHE['风险分']>=min_score]
        return df.head(50) if len(df)>0 else _RISK_CACHE.head(10)
    sampled=[]
    for p in ['000','002','300','600','601','603','605','688','920']:
        sampled.extend([c for c in ALL_STOCKS if c.startswith(p)][:30])
    results=[]
    for code in sampled:
        r=fraud.analyze(code)
        if 'error' not in r:
            sc=r['stock_code'].replace('.SH','').replace('.SZ','').replace('.BJ','')
            results.append({"代码":r['stock_code'],"名称":CODE2NAME.get(sc,"-"),"风险分":r['risk_score'],"等级":r['risk_level'],"预警数":r['alert_count'],"关键预警":", ".join(a['rule'][:6] for a in r['alerts'][:3])})
    results.sort(key=lambda x:x['风险分'],reverse=True)
    _RISK_CACHE=pd.DataFrame(results)
    df=_RISK_CACHE[_RISK_CACHE['风险分']>=min_score]
    return df.head(50)

# === Tab 4: 投资组合 ===
def pf_display():
    pf=Portfolio()
    lines=["### 我的投资组合",""]
    idx=0
    if pf.data['watchlist']:
        lines.append(f"**自选股**({len(pf.data['watchlist'])}只)\n")
        for item in pf.data['watchlist']:
            nm=f" — {item.get('name','')}" if item.get('name') and item.get('name')!=item['code'] else ""
            lines.append(f"{idx+1}. `{item['code']}`{nm}\n");idx+=1
        lines.append("")
    if pf.data['holdings']:
        lines.append(f"**持仓**({len(pf.data['holdings'])}只)\n")
        total=0
        for item in pf.data['holdings']:
            cost=item['shares']*item['cost_price'];total+=cost
            nm=f" — {item.get('name','')}" if item.get('name') and item.get('name')!=item['code'] else ""
            lines.append(f"{idx+1}. `{item['code']}`{nm}")
            lines.append(f"   {item['shares']}股 x {item['cost_price']:.2f} = **{cost/10000:.2f}万**\n");idx+=1
        lines.append(f"*总成本:**{total/10000:.2f}万***")
    if idx==0: lines.append("*暂无自选股或持仓*")
    return "\n".join(lines)

def pf_get_items():
    pf=Portfolio();items=[]
    for i in pf.data['watchlist']:items.append((i['code'],i.get('name',''),'自选'))
    for i in pf.data['holdings']:items.append((i['code'],i.get('name',''),'持仓'))
    return items

def watch_autofill(code,name):
    return auto_fill(code,name)

def pf_add_watch(code,name):
    code,name=auto_fill(code,name)
    pf=Portfolio();r=pf.add_watchlist(code,name)
    choices=[f"{t}:{c} {n}" for c,n,t in pf_get_items()]
    return r,gr.Radio(choices=choices,visible=bool(choices)),pf_display()

def pf_add_hold(code,shares,cost,name):
    code,name=auto_fill(code,name)
    if not code or not shares: return "请填写代码和股数",gr.Radio(visible=False),pf_display()
    pf=Portfolio();r=pf.add_holding(code,int(shares),float(cost or 0),name)
    choices=[f"{t}:{c} {n}" for c,n,t in pf_get_items()]
    return r,gr.Radio(choices=choices,visible=bool(choices)),pf_display()

def pf_remove_selected(sel):
    if not sel: return pf_display()
    code=sel.split(":")[1].split(" ")[0]
    pf=Portfolio();pf.remove(code)
    choices=[f"{t}:{c} {n}" for c,n,t in pf_get_items()]
    return pf_display(),gr.Radio(choices=choices,visible=bool(choices))

# === Tab 5: 事件脉络 ===
def cluster_report():
    return clusterer.cluster_report()

def event_timeline(code):
    tl=clusterer.stock_timeline(code)
    if tl['total_events']==0: return f"*{code} 无事件记录*"
    lines=[f"### {code} 事件时间线","",f"总事件:**{tl['total_events']}**条 | 风险占比:**{tl['risk_ratio']}**",f"{tl['first_event']} ~ {tl['last_event']}","",f"类型:{tl['event_types']}","","---",""]
    for e in reversed(tl['timeline'][-15:]):
        risk="!" if e['is_risk'] else " ";tags=", ".join(e['types'])
        lines.append(f"- [{risk}] **{e['date']}** [{tags}]")
        lines.append(f"  {e['title'][:90]}")
    return "\n".join(lines)

# === 构建界面 ===
with gr.Blocks(title="金融AI智能助手") as demo:
    gr.HTML("<div style='text-align:center'><h1>金融AI智能助手</h1><p style='color:#64748b'>财报反欺诈 · 股权穿透 · 研报检索 · 智能对话 · 持仓管理</p></div>")

    with gr.Tabs():
        # TAB 1
        with gr.Tab("智能对话"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot=gr.Chatbot(value=[{"role":"assistant","content":"### 欢迎!\n试试:分析 688765 财务风险 / 688765 股权结构 / 东吴证券最新研报 / 帮我看看自选股"}],height=500)
                    with gr.Row():
                        msg=gr.Textbox(placeholder="输入问题...",scale=5,container=False)
                        send=gr.Button("发送",variant="primary",scale=1)
                    clear=gr.Button("重置对话",size="sm")
                with gr.Column(scale=1):
                    gr.Markdown("### 快捷提问")
                    gr.Button("我的自选股有没有风险",size="sm").click(lambda:"帮我看看我的自选股有没有风险",None,[msg])
                    gr.Button("最新研报有哪些",size="sm").click(lambda:"今天有哪些最新研报",None,[msg])
                    gr.Button("688765 综合分析",size="sm").click(lambda:"688765 财务和股权分析",None,[msg])
            send.click(chat_fn,[msg,chatbot],[chatbot]).then(lambda:"",None,[msg])
            msg.submit(chat_fn,[msg,chatbot],[chatbot]).then(lambda:"",None,[msg])
            clear.click(lambda:(agent.memory.__init__(),[],""),None,[chatbot,msg])

        # TAB 2
        with gr.Tab("快速分析"):
            with gr.Row():
                si=gr.Textbox(label="股票代码",placeholder="输入代码如 688765",info=f"共{len(ALL_STOCKS)}只股票")
                ab=gr.Button("分析",variant="primary")
            with gr.Row():
                g1=gr.Plot(label="风险仪表盘");g2=gr.Plot(label="风险雷达")
            g3=gr.Plot(label="股权穿透(悬停看详情)")
            at=gr.Markdown("选择股票后点击分析")
            ab.click(quick_analysis,[si],[g1,g2,g3,at])

        # TAB 3
        with gr.Tab("风险排行"):
            with gr.Row():
                ms=gr.Slider(0,80,50,step=10,label="最低风险分")
                rb=gr.Button("刷新",variant="primary")
            rt=gr.Dataframe(label="高风险股票排行",max_height=600)
            rb.click(risk_ranking,[ms],[rt])
            demo.load(risk_ranking,[ms],[rt])

        # TAB 4
        with gr.Tab("投资组合"):
            with gr.Row():
                with gr.Column(scale=3):
                    pm=gr.Markdown(pf_display())
                    refresh=gr.Button("刷新",size="sm")
                with gr.Column(scale=2):
                    gr.Markdown("### 添加资产")
                    with gr.Group():
                        gr.Markdown("**添加自选**")
                        with gr.Row():wc=gr.Textbox(placeholder="600519",label="代码");wn=gr.Textbox(placeholder="贵州茅台",label="名称")
                        wb=gr.Button("添加自选",variant="secondary")
                    with gr.Group():
                        gr.Markdown("**添加持仓**")
                        with gr.Row():hc=gr.Textbox(placeholder="601688",label="代码");hn=gr.Textbox(placeholder="华泰证券",label="名称")
                        with gr.Row():hs=gr.Number(value=1000,label="股数");hp=gr.Number(value=0.0,label="成本价")
                        hb=gr.Button("添加持仓",variant="secondary")
                    am=gr.Markdown("")
                    items=pf_get_items()
                    init_choices=[f"{t}:{c} {n}" for c,n,t in items] if items else []
                    gr.Markdown("---\n**删除**(选中后点确认)")
                    dr=gr.Radio(choices=init_choices,label="选择股票",visible=bool(init_choices),interactive=True)
                    db=gr.Button("确认删除",variant="stop")
                    dm=gr.Markdown("")
            # 自动补全已在按钮点击时内置(auto_fill)，不在输入时实时触发以避免反馈循环
            def refresh_all():
                items=pf_get_items()
                choices=[f"{t}:{c} {n}" for c,n,t in items] if items else []
                return pf_display(),gr.Radio(choices=choices,visible=bool(choices))
            refresh.click(refresh_all,None,[pm,dr])
            wb.click(pf_add_watch,[wc,wn],[am,dr,pm])
            hb.click(pf_add_hold,[hc,hs,hp,hn],[am,dr,pm])
            db.click(pf_remove_selected,[dr],[pm,dr])

        # TAB 5
        with gr.Tab("事件脉络"):
            gr.Markdown("## 舆情事件聚类")
            cm=gr.Markdown(cluster_report())
            gr.Markdown("### 单股时间线")
            with gr.Row():
                es=gr.Textbox(placeholder="输入股票代码,如 603377",label="股票代码")
                eb=gr.Button("查询",variant="primary")
            et=gr.Markdown("")
            eb.click(event_timeline,[es],[et])

        # TAB 6
        with gr.Tab("关于"):
            gr.Markdown("""## 系统架构
```
用户输入 -> LLM决策 -> 工具调度(财报/股权/研报/持仓) -> LLM生成回复
```
| 模块 | 技术 |
|------|------|
| 前端 | Gradio + Plotly |
| Agent | GLM-5.2 语义理解 + 关键词路由 |
| 财报 | 12条排雷规则 + 多期趋势 |
| 股权 | BFS多跳穿透(70,221节点) |
| 研报 | 55,214篇全文检索 |
| 行情 | 腾讯财经实时API |
""")

if __name__=="__main__":
    print("\n启动: http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1",server_port=7860,share=False,theme=gr.themes.Soft(primary_hue="emerald"))
