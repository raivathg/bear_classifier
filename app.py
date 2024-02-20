from fastai.vision.all import *

def type_of_bear(x): return x[0].isupper() 
categories = ('Black bear', 'Grizzly bear', 'Teddy Bear') 
learn = load_learner('types_of_bear.pkl')
def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

#|export
import gradio as gr
# image = gr.inputs.Image(shape=(192, 192))
# label = gr.outputs.Label()
samples = ['grizzly_bear.jpg', 'teddy_bear.jpg', 'black_bear.jpg']
intf = gr.Interface(fn=classify_image, inputs="image", outputs="label", examples=samples)
intf.launch(inline=False)