import streamlit as st
import plotly
import os 
import json
from PIL import Image

st.set_page_config(page_title = 'AI Reg NLP Vis', layout = 'wide')
st.markdown("<h1 style='text-align: center; color: grey;'>AI Regulatory Document NLP Analysis</h1>", unsafe_allow_html=True)
st.markdown("""<p style='text-align: center; color: grey;'>13 documents were clustered using spectral clustering, resulting 
in N initial topics. These were fine-tuned and merged based on cosine similarity, resulting in 28 final topics.</p>""", unsafe_allow_html=True)


prefix = os.getcwd()
path = '/plotly-figs'
mapping = {'0_doc-topics-pct.json':{
    'title':'Document Composotion Percent',
    'caption':'''A view of the composition of each document by percent.
      Hover over each bar to identify the major contributors.'''
},
'2_sent-topics.json':{
    'title':'Sentences inside Topics',
    'caption':'''A fine-grained view where we can visualize 
    the sentences inside the topics to see if they were 
    assigned correctly and whether they make sense.'''
},
'3_similarity-matrix.json':{
    'title':'Topic Similarity Matrix',
    'caption':'''A matrix indicating how similar certain 
    topics are to each other by simply applying cosine
    similarities.'''
},
'1_topic-distance.json':{
    'title':'Intertopic Distance Map',
    'caption':'''A representation of the topics in 2D such that
    we can create an interactive view. The slider can select the topic which 
    then lights up red. If you hover over a topic, then general information is 
    given about the topic, including the size of the topic and its corresponding words.'''
},
'sentence-dis.json':{
    'title':'baz',
    'caption':'''Afol red. If you hover over a topic, then general information is 
    given about the topic, including the size of the topic and its corresponding words.'''
},
'doc-percent.json':{
    'title':'bar',
    'caption':'''A representation of the topics in 2D such that
    we can create an interactive view. The slider can select the topic which 
    then lights up red. If you hover over a topic, then general information is 
    given about the topic, including the size of the topic and its corresponding words.'''
},
'topic_top_words.json':{
    'title':'foo',
    'caption':'''foo
    p red. If you hover over a topic, then general information is 
    given about the topic, including the size of the topic and its corresponding words.'''
}}

st.container()
for fig in sorted(os.listdir(prefix+path)):
    st.divider()
    st.markdown(f"<h2 style='text-align: center; color: grey;'>{mapping[fig]['title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: grey;'>{mapping[fig]['caption']}</p>", unsafe_allow_html=True)
    with open(prefix+path+'/'+fig) as f:
        data = json.load(f)
        st.plotly_chart(plotly.io.from_json(data).update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            # paper_bgcolor="LightSteelBlue",
            # width=1400,
            # height=800,
            # font=dict(size=10, color='DarkSlateGray'),
            title={
            'text': f"<b>{mapping[fig]['title']}<b>",
            'text': f"",
            'y':0.5,
            # 'x':0.5,
            # 'xanchor': 'center',
            'yanchor': 'top', 
            # 'font':dict(size=23),
            'font_color': "black",},
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
        
image = Image.open('deloitte-logo-white.png')
st.sidebar.image(image)
st.sidebar.title("RegGPT V0.0.1")
