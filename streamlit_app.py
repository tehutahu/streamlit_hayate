import io
from urllib import request

from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms


CLASS_NAMES = ['official', 'user']
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def build_model():
    model = torchvision.models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    m_state_dict = torch.load('models/hayate_finetune.pt', map_location=DEVICE)
    model.load_state_dict(m_state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def inference(model, image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_t = preprocess(image)
    batch_t = torch.unsqueeze(image_t, 0)
    batch_t = batch_t.to(DEVICE)
    
    out = model(batch_t)
    _, index = torch.max(out, 1)
    per = nn.functional.softmax(out, dim=1)[0] * 100
    return CLASS_NAMES[index[0]], per[index[0]].item()

class Body(object):
    def __init__(self, container: st.container, title: str, **sidebar) -> None:
        self.body = container
        self.sidebar = sidebar
        self.input_container = None
        self.setting_container = None
        self.running_container = None
        self.area_funcs = [self.input_area, self.setting_area, self.running_area]
        self.make_body(title)

    def make_body(self, title: str) -> None:
        with self.body:
            st.title(title)
            self.input_container = st.container()
            st.write('---')
            self.setting_container = st.container()
            st.write('---')
            self.running_container = st.container()

    def input_area(self) -> None:
        with self.input_container:
            st.write('This is input area')
    
    def setting_area(self) -> None:
        with self.setting_container:
            st.write('This is setting area')
    
    def running_area(self) -> None:
        with self.running_container:
            st.write('This is running area')

    def display_area(self) -> None:
        for f in self.area_funcs:
            f()


class Vision(Body):
    def __init__(self, container: st.container, title: str, model: torch.nn.Module, **sidebar) -> None:
        super().__init__(container, title, **sidebar)
        self.input_data = None
        self.model = model
        self.setting = {
            'color': 'green',
            'font_path': './Helvetica 400.ttf',
            'font_size': 15
        }

    def input_area(self) -> None:
        with self.input_container:
            col1, col2 = st.columns([3, 1])
            with col2:
                option = st.selectbox(
                    '入力形式', 
                    ('URL', 'file')
                )
            with col1:
                if option == 'file':
                    err = st.empty()
                    uploaded_file = st.file_uploader('画像ファイルを入力してください', ['png', 'jpg', 'jpeg'])
                    self.input_data = uploaded_file
                else:
                    self.input_data = st.text_area(
                        '画像のURLを入力してください',
                        'https://imas.gamedbs.jp/cgss/images/F5uD5HHnT0IeMsl0dohw1SSMRZhyiaMqkwB7YcGB0Vc.jpg?1447915675'
                    )
        return None

    def setting_area(self) -> None:
        # with self.setting_container:
        #     st.subheader('実行設定')
        #     col1, col2, col3 = st.columns(3)
        #     with col1:
        #         self.setting['color'] = st.selectbox(
        #             'box color',
        #             ('green', 'red', 'blue')
        #         )
        #     with col2:
        #         self.setting['font_color'] = st.selectbox(
        #             'font color',
        #             ('white', 'black', 'gray')
        #         )
        #     with col3:
        #         self.setting['font_size'] = st.slider('font size', 5, 20, 12)
        return None

    def running_area(self) -> None:
        if self.input_data is None:
            return None
        else:
            with self.running_container:
                st.header('入力データ')
                st.image(self.input_data)
                if st.button('Run'):
                    if isinstance(self.input_data, str):
                        try:
                            response = request.urlopen(self.input_data)
                            content = response.read()
                            image = Image.open(io.BytesIO(content))
                        except Exception as e:
                            st.error('有効な画像のURLを入力してください')
                            st.stop()
                    else:
                        image = Image.open(self.input_data)
                    label, per = inference(self.model, image)
                    st.write('---')
                    st.header('結果')
                    st.subheader('Predict: {}, Percentage: {:.2f}'.format(label, per))
                    st.image(image)
        return None
        
def main():
    model = build_model()
    body_container = st.container()
    body = Vision(body_container, 'Check hayate whether official or not?', model)
    body.display_area()


if __name__ == '__main__':
    main()