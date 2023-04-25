import streamlit as st
import plotly
import os 
import json
import pandas as pd
from PIL import Image
from bertopic import BERTopic
import networkx as nx
from bokeh.models import (WheelPanTool, BoxZoomTool, ResetTool,  Circle, HoverTool,
                          MultiLine, NodesAndLinkedEdges, Plot, TapTool, WheelZoomTool)
from bokeh.plotting import from_networkx, show, output_file, save, figure

prefix = os.getcwd()
path = '/figs'
st.set_page_config(page_title = 'AI Reg NLP Vis', layout = 'wide')

def write_h1(text:str):
    return st.markdown(f"<h1 style='text-align: center; color: grey;'>{text}</h1>", unsafe_allow_html=True)

def write_h2(text:str):
    return st.markdown(f"<h2 style='text-align: center; color: grey;'>{text}</h2>", unsafe_allow_html=True)

def write_c(text:str):
    return st.markdown(f"<p style='text-align: center; color: grey;'>{text}</p>", unsafe_allow_html=True)

def draw_plotly(file:json):
    with open(prefix+path+'/'+file) as f:
        data = json.load(f)
        st.plotly_chart(plotly.io.from_json(data).update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            # paper_bgcolor="LightSteelBlue",
            # width=1400,
            # height=800,
            font=dict(size=10, color='Black'),
            title={
            'text': ""},
            # 'text': f"<b>{mapping[fig]['title']}<b>",
            # 'text': f"",
            # 'y':0.5,
            # 'x':0.5,
            # 'xanchor': 'center',
            # 'yanchor': 'top', 
            # 'font':dict(size=23),
            # 'font_color': "black",},
        #     xaxis = dict(
        #     # tickmode = 'linear',
        #     tickfont = dict(size=9)
        #     ),
        #     yaxis = dict(
        #    tickmode = 'linear',
        #    tickfont = dict(size=9)
        #    )),
        ),
        #    width=1100,
           use_container_width=True,)
        return
    


#### TITLE SECTION
write_h1("Unlocking Insights: NLP Analysis of AI Regulatory Documents")
write_c("""
A Comprehensive Guide to Enhancing AI Regulation through Natural Language Processing
""")

#### PREPROCESSING SECTION
st.divider()
write_h2("Preprocessing: Similarity Matrix & Inter Topic Distance")

write_c('''13 documents were clustered using spectral clustering, resulting 
in 31 preliminary topics. These were fine-tuned and merged based on cosine similarity, resulting in 28 final topics. 
We can ensure minimal overlap of our topics by inspecting the intertopic distance and a similarity matrix.
The intertopic distance map uses UMAP dimensionality reduction to produce a 2D, interpretable view of the topics and 
the topic similarity matrix indicates how similar certain topics are to each other by applying cosine similarities''')


c1, c2 = st.columns(2)
with c1:
    draw_plotly('3_similarity-matrix.json')
with c2:
    draw_plotly('1_topic-distance.json')

#### TOPIC ANALYSIS SECTION
st.divider()
write_h2('Topic Analysis: Common Words & Network Graph')
write_c('''Once topics were identified we can understand what they are and how they relate to eachother - key topic: Pros & Cons of using AI in FS''') 

c3, c4 = st.columns(2)
with c3:
    topic_model = BERTopic.load("my_model3")
    topic_dict = {3: "Risk management framework for AI",
                4: "Impact of AI in modern society",
                5: "Regulatory requirements for High risk AI",
                6: "Discrimination and bias in AI models",
                7: "Assessing and mitigating risks related to user rights",
                8: "AI regulatory landscape across different jurisdictions",
                9: "Processing personal data (fair, safe, compliant)",
                10: "User protection against AI",
                2: "Outlier",
                11: "Implementing data protection controls",
                12: "Biometric identification",
                0: "Pros and cons of using AI in financial services",
                1: "Regulation of intermediary services",
                13: "ML definition",
                14: "Explainability, transparency and interpretability of AI",
                15: "Principles for designing responsible AI",
                16: "Ethical assessment in the AI development lifecycle",
                17: "AIDA and high impact AI systems",
                18: "Testing and monitoring AI systems",
                19: "Role of EU commission",
                20: "Developing AI models",
                21: "Human review and intervention",
                22: "Enforcing AI regulation and penalties" ,
                23: "Role of digital service coordinators",
                    24: "Data governance",
                    25: "AI security and 3rd part AI risks",
                    26: "Implementing an AI governance Framework",
                    27: "AI compliance",
                    28: "AI Compliance controls and obligations in EU"}
    topic_model.set_topic_labels(topic_dict)
    f = topic_model.visualize_barchart(top_n_topics=12, n_words=4,custom_labels=True)
    st.plotly_chart(f.update_layout(title=dict(text='')), user_container_width=True)

with c4:
    df = pd.read_csv('completed_v4.csv')
    df.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis =1, inplace = True)
    df.columns = ['Sentence', 'Document ID', 'Topic']

    grouped_df = df.groupby(['Document ID' ,'Topic']).count().reset_index()
    G = nx.from_pandas_edgelist(grouped_df, 'Topic','Document ID', ['Sentence'])

    degrees = dict(nx.degree(G))
    for k in degrees.keys():
        degrees[k] = (degrees[k] - 5)**1.2

    # set attrs for colure, hierachy and size
    nx.set_node_attributes(G, name='degree', values=degrees)
    nx.set_node_attributes(G,name='Topic', values=topic_dict)


    plot = Plot()
    # plot.title.text = "(hover to see detailed information)"
    plot.add_tools(BoxZoomTool(), ResetTool(), TapTool())

    # render bokeh graph
    graph_renderer = from_networkx(G, nx.spring_layout)
    # default node color
    graph_renderer.node_renderer.glyph = Circle(size='degree', fill_color='#30B6E6')

    # node highlight formatting
    graph_renderer.node_renderer.selection_glyph = Circle(size='degree', fill_color='#075e67', line_width='1')
    graph_renderer.node_renderer.hover_glyph = Circle(size='degree', fill_color='#075e67', line_width='1')

    graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width='1')
    # edge highlight formatting
    graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color='#1d1d1f', line_width='1')
    graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color='#1d1d1f', line_width='1')
    graph_renderer.edge_renderer.data_source.data["line_width"] = [G.get_edge_data(a,b)['Sentence']**0.33 for a, b in G.edges()]
    graph_renderer.edge_renderer.glyph.line_width = {'field': 'line_width'}



    # interaction hover allow
    graph_renderer.selection_policy = NodesAndLinkedEdges()
    graph_renderer.inspection_policy = NodesAndLinkedEdges()
    plot.renderers.append(graph_renderer)
    plot.outline_line_color=None
    node_hover_tool = HoverTool(tooltips=[
        ("Topic Name", "@Topic"),
        ("Topic ID", "@index"),
    ])
    plot.add_tools(node_hover_tool,WheelPanTool(), WheelZoomTool())
    st.bokeh_chart(plot, use_container_width=True)
    # st.components.v1.html(html_data, use_container_width=True)#height=1400, scrolling=False)
    

#### DOCUMENT & TOPIC COMPOSITION 
write_h2('Document & Topic Composition')
write_c('''Finally, after understanding each topic, we can explore the composition of each document and compare the sentences that contribute
to each topic.''') 

c5, c6 = st.columns(2)
with c5:
    draw_plotly('doc-percent.json')
with c6:
    draw_plotly('sentence-dis.json')

### old stuff
# mapping = {'0_doc-topics-pct.json':{
#     'title':'Document Composition Percent',
#     'caption':'''A view of the composition of each document by percent.
#       Hover over each bar to identify the major contributors.'''
# },
# '2_sent-topics.json':{
#     'title':'Sentences inside Topics',
#     'caption':'''A fine-grained view where we can visualize 
#     the sentences inside the topics to see if they were 
#     assigned correctly and whether they make sense.'''
# },
# '3_similarity-matrix.json':{
#     'title':'Topic Similarity Matrix',
#     'caption':'''A matrix indicating how similar certain 
#     topics are to each other by simply applying cosine
#     similarities.'''
# },
# '1_topic-distance.json':{
#     'title':'Intertopic Distance Map',
#     'caption':'''A representation of the topics in 2D such that
#     we can create an interactive view. The slider can select the topic which 
#     then lights up red. If you hover over a topic, then general information is 
#     given about the topic, including the size of the topic and its corresponding words.'''
# },
# 'sentence-dis.json':{
#     'title':'baz',
#     'caption':'''Afol red. If you hover over a topic, then general information is 
#     given about the topic, including the size of the topic and its corresponding words.'''
# },
# 'doc-percent.json':{
#     'title':'bar',
#     'caption':'''A representation of the topics in 2D such that
#     we can create an interactive view. The slider can select the topic which 
#     then lights up red. If you hover over a topic, then general information is 
#     given about the topic, including the size of the topic and its corresponding words.'''
# },
# 'topic_top_words.json':{
#     'title':'foo',
#     'caption':'''foo
#     p red. If you hover over a topic, then general information is 
#     given about the topic, including the size of the topic and its corresponding words.'''
# }}

# st.container()
# for fig in sorted(os.listdir(prefix+path)):
#     st.divider()
#     st.markdown(f"<h2 style='text-align: center; color: grey;'>{mapping[fig]['title']}</h1>", unsafe_allow_html=True)
#     st.markdown(f"<p style='text-align: center; color: grey;'>{mapping[fig]['caption']}</p>", unsafe_allow_html=True)
#     with open(prefix+path+'/'+fig) as f:
#         data = json.load(f)
#         st.plotly_chart(plotly.io.from_json(data).update_layout(
#             margin=dict(l=20, r=20, t=20, b=20),
#             # paper_bgcolor="LightSteelBlue",
#             # width=1400,
#             # height=800,
#             # font=dict(size=10, color='DarkSlateGray'),
#             title={
#             'text': f"<b>{mapping[fig]['title']}<b>",
#             'text': f"",
#             'y':0.5,
#             # 'x':0.5,
#             # 'xanchor': 'center',
#             'yanchor': 'top', 
#             # 'font':dict(size=23),
#             'font_color': "black",},
#         #     xaxis = dict(
#         #     # tickmode = 'linear',
#         #     tickfont = dict(size=9)
#         #     ),
#         #     yaxis = dict(
#         #    tickmode = 'linear',
#         #    tickfont = dict(size=9)
#         #    )),
#         ),
#         #    width=1100,
#            use_container_width=True,)
        
image = Image.open('deloitte-logo-white.png')
st.sidebar.image(image)
st.sidebar.title("RegGPT V0.0.1")
